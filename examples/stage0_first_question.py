"""第一版の研究課題の縦断デモ (設計書 §27, Stage 0-2, §30)。

問い: 総配合を保ったまま供給配置を変更したとき、測定可能な五元素反応領域の
形成時期・寸法・再現性はどう変わるか。また、その変化は供給距離・界面抵抗・
先行反応のどれで説明されるか。

比較: R(基準・無作為混合) / P1(距離制御) / P2(接触順序制御)。

注意: 本スクリプト中の物性値・寸法・効用は例示値であり、設計書 §0 の通り
実験条件・物性値を確定しない。実研究では校正データ・実測で置換すること。

実行: python3 examples/stage0_first_question.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # type: ignore

from sdm_react.alternatives import (
    ConsistencySpec,
    RivalHypothesis,
    RivalPrediction,
    consistent_hypotheses,
    discriminate,
    mixture_prediction,
)
from sdm_react.core import ELEMENTS
from sdm_react.goals import Criterion, GoalCompiler, InformationType, TargetSpec
from sdm_react.inventory import InventoryLedger
from sdm_react.observation import (
    CoolingClass,
    CoolingHistory,
    MeasurementCondition,
    ObservationModel,
    apply_cooling,
)
from sdm_react.pipeline import Pipeline, default_ticket_for
from sdm_react.planner import (
    ActionKind,
    ActionRole,
    ExperimentPlanner,
    FabConstraints,
    UtilityVector,
    check_fabricable,
    pareto_front,
)
from sdm_react.process import (
    Arrangement,
    FabDesign,
    InitialStateCandidate,
    ProcessState,
)
from sdm_react.reachability import (
    BoundBasis,
    TransportCap,
    assess_inventory,
    assess_transport,
)
from sdm_react.reaction import (
    M1EffectiveReaction,
    M1Law,
    M1Params,
    entry_time,
    passing_intervals,
    robust_intersection,
)
from sdm_react.scope import Mechanism, ScopeAudit

print("=" * 72)
print("SDM-REACT Stage 0-2 デモ: 総配合固定・供給配置変更の効果 (§27)")
print("=" * 72)

# ---------------------------------------------------------------- ① 目標定義
compiler = GoalCompiler()
targets = [
    TargetSpec(
        id="T-Q1", information_type=InformationType.Q1,
        scientific_question="測定可能な五元素反応領域はいつ・どれだけ形成されるか",
        material_state_required=["五元素を含む反応領域", "観測可能な寸法"],
        minimum_observation_requirements=["五元素の定量", "分析体積より大きい領域"],
        required_evidence=["局所組成", "組織尺度"],
        competing_interpretations=["微細混合の平均化", "冷却変態"],
        success_criteria=[
            Criterion("q1-nelem", "定量元素数", "n_elements_quantified", 5.0, ">="),
            Criterion("q1-size", "領域寸法[um]", "region_um", 3.0, ">="),
        ],
        failure_criteria=[Criterion("q1-depl", "供給枯渇", "min_inventory_mol", 0.0, "<=")],
        inconclusive_conditions=["冷却影響が未評価 (C4) の場合は高温帰属を保留"],
    ),
    TargetSpec(
        id="T-Q6", information_type=InformationType.Q6,
        scientific_question="配置効果の主因は供給距離/界面抵抗/先行反応/観測混合のどれか",
        competing_interpretations=["H1:供給距離", "H2:界面抵抗", "H3:先行反応捕捉", "H4:観測混合"],
        success_criteria=[Criterion("q6-sep", "仮説分離", "worst_separation", 0.0, ">=")],
    ),
]
compiled = [compiler.compile(t) for t in targets]
print(f"\n[①] 目標を分解: {[c.target.id for c in compiled]}")

# ---------------------------------------------------------------- ② 監査
audit = ScopeAudit()
audit.observe(Mechanism.SOLID_DIFFUSION, "予備観察で反応層を確認 (例示)")
audit.mark_outside_model(Mechanism.LIQUID_FORMATION, "現モデルは液相を扱わない")
print(f"[②] 分岐判断: {audit.branch(rank_sensitive=[Mechanism.GB_TRANSPORT])['decision']}")

# ---------------------------------------------------------------- ③ 候補設計
TOTAL = {"Al": 1e-3, "Si": 1e-3, "Ti": 1e-3, "Fe": 1e-3, "Cu": 1e-3}  # 総配合は固定
SIZE = {"Al": 20.0, "Si": 20.0, "Ti": 20.0, "Fe": 20.0, "Cu": 20.0}
designs = [
    FabDesign("D-R", dict(TOTAL), dict(SIZE), Arrangement.R,
              {"min_feature_um": 20.0, "size_mm": 5.0}, operable_vars=["arrangement"]),
    FabDesign("D-P1", dict(TOTAL), dict(SIZE), Arrangement.P1,
              {"min_feature_um": 50.0, "size_mm": 5.0,
               "note": "Cu供給源を遠方化 (例示)"}, operable_vars=["arrangement"]),
    FabDesign("D-P2", dict(TOTAL), dict(SIZE), Arrangement.P2,
              {"min_feature_um": 50.0, "size_mm": 5.0,
               "note": "Al-Si先接触後にFe/Cu供給 (例示)"}, operable_vars=["arrangement"]),
]
cons = FabConstraints(arrangement_resolution_um=10.0, max_sample_mm=10.0)
for d in designs:
    ok, reasons = check_fabricable(d.arrangement, d.arrangement_detail, cons,
                                   fab_error_um=5.0)
    print(f"[③] {d.id} ({d.arrangement.value}): 作製可能={ok} {reasons}")

# ------------------------------------------------------------ ④ 初期状態候補
ps = ProcessState()
ensembles = {}
# 例示の供給源間距離: R=30um, P1(Cu遠方)=120um, P2=60um
dist = {"D-R": 30.0, "D-P1": 120.0, "D-P2": 60.0}
for d in designs:
    cands = [InitialStateCandidate(
        id=f"{d.id}-c{i}", residual_inventory_mol=dict(TOTAL),
        contact_types=["Al-Si", "Ti-Fe", "Cu-mix"],
        source_distances_um={"Cu_to_zone": dist[d.id]},
        source="sister_sample(例示)",
    ) for i in range(2)]
    ensembles[d.id] = ps.predict_initial(d, cands)
    print(f"[④] {d.id}: 初期候補 {len(cands)} 件 (無重み集合, kind=pre_fab)")

# -------------------------------------------------------- ⑤ 到達可能性評価
req = [{el: 1e-6 for el in ELEMENTS}]  # 例示の要求量
for d in designs:
    rep = assess_inventory(req, dict(TOTAL), assumptions=["閉鎖系", "例示要求量"],
                           uncovered_mechanisms=["液相経路は未包含"])
    print(f"[⑤] {d.id}: 在庫評価={rep.verdict.value}")
trep = assess_transport(
    {"Cu": 1e-6},
    [TransportCap("cu_src", "zone", "Cu", 5e-6, BoundBasis.CONSERVATIVE, "例示上限")],
    target_region="zone", source_regions=["cu_src"], assumptions=["緩和問題"])
print(f"[⑤] Cu輸送の外側評価={trep.verdict.value} (実現保証ではない)")

# ------------------------------------------------ ⑥ 有効反応予測 (M1)
times = [0, 900, 3600, 10800, 36000]
# H1(供給距離) vs H2(界面抵抗) を成長係数のシナリオで表現 (例示値)
scen = {"H1-distance": {"D-R": 4e-14, "D-P1": 1e-14, "D-P2": 2.5e-14},
        "H2-interface": {"D-R": 2e-14, "D-P1": 1.8e-14, "D-P2": 1.2e-14}}
xi_um: dict = {}
for s_name, kvals in scen.items():
    xi_um[s_name] = {}
    for d in designs:
        m = M1EffectiveReaction(M1Params(
            law=M1Law.PARABOLIC, K_eff_m2_s=kvals[d.id],
            t0_definition="保持開始を反応開始と定義 (例示)",
            product_composition={el: 0.2 for el in ELEMENTS},
            product_molar_volume_m3_mol=1e-5, contact_area_m2=1e-8,
            domain={"note": "例示域"}))
        out = m.predict({"time_s": times, "available_mol": {el: 1e-6 for el in ELEMENTS}})
        xi_um[s_name][d.id] = [x * 1e6 for x in out.value["xi_m"]]
        print(f"[⑥] {s_name} {d.id}: xi[um]={['%.2f' % v for v in xi_um[s_name][d.id]]}")

# Cu 参加時刻 (例示の在庫履歴から)
for d in designs:
    hist = [0.0, 2e-7, 6e-7, 9e-7, 1e-6]
    e = entry_time("Cu", times, hist, 5e-7, "定量下限からの観測的根拠 (例示)", "定量可能性")
    print(f"[⑥] {d.id}: Cu t_entry={e.t_entry_s}s")

# ------------------------------------------------------ ⑦ 冷却・仮想測定
cooling = CoolingHistory(time_s=[0, 600], temperature_K=[1000, 300],
                          cooling_class=CoolingClass.C4)
cond = MeasurementCondition(
    id="EPMA-demo", probe_volume_um3=1.0,
    detection_limit={el: 0.01 for el in ELEMENTS},
    noise_sigma={el: 0.005 for el in ELEMENTS})
obs = ObservationModel(cond)
NOMINAL = {"Al": 0.25, "Si": 0.15, "Ti": 0.2, "Fe": 0.2, "Cu": 0.2}  # 例示の真組成
for d in designs:
    st = apply_cooling({"xi_um": xi_um["H1-distance"][d.id][-1]}, cooling)
    o = obs.predict({"state_composition": NOMINAL, "seed": 42})
    nq = sum(1 for v in o.value["point"].quantification.values() if v == "pass")
    j = compiler.judge(targets[0], {"n_elements_quantified": float(nq),
                                    "region_um": st["xi_um"]})
    print(f"[⑦] {d.id}: 定量元素={nq}/5, xi={st['xi_um']:.2f}um → T-Q1={j.overall} "
          f"({st.get('high_T_claim_restriction', '')[:20]}...)")

# 時間窓: xi≥3um を合格として T_q を抽出
windows = {}
for s_name in scen:
    for d in designs:
        verd = ["pass" if v >= 3.0 else "fail" for v in xi_um[s_name][d.id]]
        windows[f"{s_name}/{d.id}"] = passing_intervals(times, verd)
print(f"[⑦] 時間窓 T_q(例): {windows}")
print(f"[⑦] 頑健窓 ∩(H1,H2)/D-R = {robust_intersection({'H1': windows['H1-distance/D-R'], 'H2': windows['H2-interface/D-R']})}")

# -------------------------------------------------- ⑧ 対立説明と識別
spec = ConsistencySpec("max_abs_deviation", 0.03, registered_before_observation=True)
rivals = [
    RivalPrediction(RivalHypothesis.UNIFORM_SINGLE_PHASE, dict(NOMINAL)),
    mixture_prediction([{"Al": 0.5, "Si": 0.3, "Ti": 0.2, "Fe": 0.0, "Cu": 0.0},
                        {"Al": 0.0, "Si": 0.0, "Ti": 0.2, "Fe": 0.4, "Cu": 0.4}],
                       [0.5, 0.5]),
]
hits = consistent_hypotheses(NOMINAL, rivals, spec)
print(f"[⑧] 整合する説明: {[h.value for h in hits]} (単相↔混合は要識別)")
drep = discriminate("追加TEM回折", {"uniform": {"Al": (0.24, 0.26)},
                                    "mixture": {"Al": (0.20, 0.30)}},
                    "Al", 0.005, cost=5.0)
print(f"[⑧] {drep.recommendation}")

# ------------------------------------------------ ⑨⑩ 情報比較と選択
planner = ExperimentPlanner()
utils = {
    "D-R": UtilityVector(2.0, 1.0, 1.0, 2.0, -1.0, definitions="例示値 (要固定)"),
    "D-P1": UtilityVector(3.0, 1.5, 3.0, 1.0, -2.0, definitions="例示値 (要固定)"),
    "D-P2": UtilityVector(2.5, 2.0, 2.5, 1.5, -2.0, definitions="例示値 (要固定)"),
}
print(f"[⑨] Pareto前線: {pareto_front(utils)} (単一総合点は使わない)")
sel = planner.select(utils, {"H1": {"D-R": 2.0, "D-P1": 5.0, "D-P2": 4.0},
                              "H2": {"D-R": 4.0, "D-P1": 2.0, "D-P2": 3.0}},
                     probabilistic=False)
print(f"[⑩] 推奨={sel.recommended}, 代替={sel.alternatives}")
for r in sel.undetermined_reasons:
    print(f"     判断不能理由: {r}")
print(f"     役割: {sel.roles}")

# ------------------------------------------------------ ⑪ 実験前予測の固定
ticket = default_ticket_for("ACT-001", ["T-Q1", "T-Q6"], "discrimination",
                            "arrangement_change")
ticket.modified_controls = ["arrangement: R→P1 (Cu供給源の遠方化)"]
ticket.fixed_conditions = ["総配合", "焼結履歴", "焼鈍温度履歴", "冷却履歴", "測定条件A"]
ticket.model_ids = ["M1:effective_reaction", "observation:H_A"]
ticket.predicted_observations = ["P1はRより反応層形成が遅延 (H1)", "遅延なしならH2/H3を検討"]
ticket.required_measurements = ["焼結後確認", "時間依存確認", "独立作製反復"]
ticket.competing_hypotheses = ["H1", "H2", "H3", "H4"]
ticket.next_step_if_supported = "距離依存の定量化へ"
ticket.next_step_if_rejected = "界面抵抗・先行反応の識別実験へ"
ticket.limitations = ["液相は対象外", "冷却はC4 (高温帰属を制限)", "物性は例示値"]
ticket.preregister()
out_path = Path(__file__).resolve().parents[1] / "action_ACT-001.yaml"
out_path.write_text(yaml.safe_dump(ticket.to_dict(), allow_unicode=True,
                                   sort_keys=False), encoding="utf-8")
print(f"[⑪] 行動票を固定: {out_path.name} ({ticket.preregistration_timestamp})")
print("\n[⑫] 次段階: 未使用条件 (別保持時間・別バッチ) での独立検証 (§25.5)")
print("=" * 72)
print("デモ終了。数値はすべて例示であり実験検証済みではない (設計書 §0)。")
