"""ExperimentPlanner: 行動候補の比較と選択 (設計書 §16, §17, §20, §23.3)。

- 単一総合点を既定値にしない。多軸 U(d) (§20.2)
- 根拠ある確率がなければシナリオ別・順位安定性・優越・最悪条件・最大後悔 (§20.4)
- 選択アルゴリズム 12 手順 (§20.5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sdm_react.process import Arrangement


class ActionRole(str, Enum):
    IMPROVEMENT = "improvement"
    DISCRIMINATION = "discrimination"
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"


class ActionKind(str, Enum):
    """行動の種類 (§20.1)。"""

    FAB_CHANGE = "fab_change"
    ARRANGEMENT_CHANGE = "arrangement_change"
    THERMAL_CHANGE = "thermal_change"
    COOLING_CHANGE = "cooling_change"
    POST_SINTER_CHECK = "post_sinter_check"
    EXTRA_COMPOSITION = "extra_composition"
    STRUCTURE_MEASURE = "structure_measure"
    OBS_CALIBRATION = "obs_calibration"
    PROP_CALIBRATION = "prop_calibration"
    DETAIL_CALC = "detail_calc"
    CANDIDATE_REPRO = "candidate_repro"
    STOP = "stop"


@dataclass
class UtilityVector:
    """多軸評価 U(d) = (ΔI_comp, ΔI_phase, ΔI_mech, R, -C) (§20.2)。

    各指標の具体的定義はキャンペーン開始時に固定する。
    """

    dI_composition: float = 0.0
    dI_phase: float = 0.0
    dI_mechanism: float = 0.0
    robustness: float = 0.0
    neg_cost: float = 0.0  # -C
    definitions: str = ""

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.dI_composition, self.dI_phase, self.dI_mechanism,
                self.robustness, self.neg_cost)


def dominates(a: UtilityVector, b: UtilityVector) -> bool:
    """a が b を全軸で上回り少なくとも一軸で厳密に上回る (優越, §20.4)。"""
    ta, tb = a.as_tuple(), b.as_tuple()
    return all(x >= y for x, y in zip(ta, tb)) and any(x > y for x, y in zip(ta, tb))


def pareto_front(vecs: Mapping[str, UtilityVector]) -> List[str]:
    """明確に劣る候補を除いた集合 (§20.5 手順9)。"""
    ids = list(vecs)
    return [i for i in ids if not any(dominates(vecs[j], vecs[i]) for j in ids if j != i)]


def minimax_regret(
    scenario_utils: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """最大後悔 (確率根拠なしの場合, §20.4)。

    scenario_utils[scenario][candidate] = 単軸効用。
    後悔 = シナリオ内最大 - 自候補。最大後悔が小さい候補が頑健。
    """
    regrets: Dict[str, float] = {}
    cands = {c for s in scenario_utils.values() for c in s}
    for c in cands:
        worst = 0.0
        for s, utils in scenario_utils.items():
            best = max(utils.values())
            worst = max(worst, best - utils.get(c, 0.0))
        regrets[c] = worst
    return regrets


def rank_stability(
    scenario_orders: Mapping[str, List[str]],
) -> Dict[str, Tuple[int, int]]:
    """順位安定性: 候補別の (最良順位, 最悪順位)。逆転があれば単一推奨を避ける。"""
    cands = {c for o in scenario_orders.values() for c in o}
    out: Dict[str, Tuple[int, int]] = {}
    for c in cands:
        ranks = [o.index(c) for o in scenario_orders.values() if c in o]
        out[c] = (min(ranks), max(ranks)) if ranks else (-1, -1)
    return out


# ---------------------------------------------------------------------------
# §16 反応配置の設計: 作製可能性制約
# ---------------------------------------------------------------------------


@dataclass
class FabConstraints:
    """作製可能性制約 (§16.2)。"""

    available_feeds: List[str] = field(default_factory=list)
    arrangement_resolution_um: float = 0.0
    min_handling_mol: float = 0.0
    max_sample_mm: float = 0.0
    max_temperature_K: float = 0.0
    notes: List[str] = field(default_factory=list)


def check_fabricable(
    arrangement: Arrangement,
    detail: Mapping[str, Any],
    constraints: FabConstraints,
    fab_error_um: float = 0.0,
) -> Tuple[bool, List[str]]:
    """作製可能性の検査。作製誤差で構造維持できない候補は理想配置のまま推奨しない。"""
    reasons: List[str] = []
    ok = True
    feature = float(detail.get("min_feature_um", float("inf")))
    if feature < constraints.arrangement_resolution_um:
        ok = False
        reasons.append(
            f"最小構造 {feature} um < 配置分解能 {constraints.arrangement_resolution_um} um"
        )
    if fab_error_um > 0 and feature < 2.0 * fab_error_um:
        ok = False
        reasons.append(
            f"作製誤差 {fab_error_um} um に対し構造 {feature} um が維持できない (§16.2)"
        )
    size = float(detail.get("size_mm", 0.0))
    if constraints.max_sample_mm and size > constraints.max_sample_mm:
        ok = False
        reasons.append("試料寸法超過")
    return ok, reasons


# ---------------------------------------------------------------------------
# §17 操作可能性の評価
# ---------------------------------------------------------------------------


def response_matrix(
    outputs: Sequence[str],
    controls: Sequence[str],
    jac: Sequence[Sequence[float]],
    output_scales: Sequence[float],
    control_ranges: Sequence[float],
) -> Dict[str, Any]:
    """局所応答行列 B = dz/du の診断 (§17.1, §17.3)。

    単位・操作可能幅・測定誤差で規格化する。階数の高さを被覆証明としない。
    相・領域が消失する設計比較では有限介入の結果集合を比較すること (§17.2)。
    """
    try:
        import numpy as np  # type: ignore

        J = np.array(jac, dtype=float)
        so = np.array(output_scales, dtype=float)
        cu = np.array(control_ranges, dtype=float)
        Bn = (J / so[:, None]) * cu[None, :]
        sv = np.linalg.svd(Bn, compute_uv=False)
        rank = int((sv > 1e-9 * (sv.max() if sv.size else 1.0)).sum())
        return {
            "normalized": Bn.tolist(),
            "singular_values": sv.tolist(),
            "rank": rank,
            "warn": "階数の高さを五元組成空間全域の被覆証明とはしない (§17.3)",
        }
    except ImportError:
        return {
            "normalized": list(jac),
            "singular_values": [],
            "rank": -1,
            "warn": "numpyなしのため特異値未計算。被覆証明には使わない (§17.3)",
        }


# ---------------------------------------------------------------------------
# §23.3 行動票
# ---------------------------------------------------------------------------


@dataclass
class ActionTicket:
    id: str
    target_ids: List[str] = field(default_factory=list)
    role: ActionRole = ActionRole.EXPLORATION
    kind: ActionKind = ActionKind.FAB_CHANGE
    modified_controls: List[str] = field(default_factory=list)
    fixed_conditions: List[str] = field(default_factory=list)
    fabrication_constraints: List[str] = field(default_factory=list)
    initial_state_prediction: str = ""
    model_ids: List[str] = field(default_factory=list)
    predicted_observations: List[str] = field(default_factory=list)
    uncertainty_representation: str = ""
    required_measurements: List[str] = field(default_factory=list)
    sampling_plan: str = ""
    success_criteria: List[str] = field(default_factory=list)
    rejection_criteria: List[str] = field(default_factory=list)
    inconclusive_conditions: List[str] = field(default_factory=list)
    competing_hypotheses: List[str] = field(default_factory=list)
    next_step_if_supported: str = ""
    next_step_if_rejected: str = ""
    limitations: List[str] = field(default_factory=list)
    preregistration_timestamp: str = ""

    def preregister(self) -> "ActionTicket":
        """実験前予測の固定 (§20.5 手順11, §23.3)。"""
        self.preregistration_timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": {
                "id": self.id,
                "target_ids": self.target_ids,
                "role": self.role.value,
                "kind": self.kind.value,
                "modified_controls": self.modified_controls,
                "fixed_conditions": self.fixed_conditions,
                "fabrication_constraints": self.fabrication_constraints,
                "initial_state_prediction": self.initial_state_prediction,
                "model_ids": self.model_ids,
                "predicted_observations": self.predicted_observations,
                "uncertainty_representation": self.uncertainty_representation,
                "required_measurements": self.required_measurements,
                "sampling_plan": self.sampling_plan,
                "success_criteria": self.success_criteria,
                "rejection_criteria": self.rejection_criteria,
                "inconclusive_conditions": self.inconclusive_conditions,
                "competing_hypotheses": self.competing_hypotheses,
                "next_step_if_supported": self.next_step_if_supported,
                "next_step_if_rejected": self.next_step_if_rejected,
                "limitations": self.limitations,
                "preregistration_timestamp": self.preregistration_timestamp,
            }
        }


# ---------------------------------------------------------------------------
# §20.5 選択アルゴリズム
# ---------------------------------------------------------------------------


@dataclass
class CandidateScore:
    id: str
    utility: UtilityVector
    scenario_utils: Dict[str, float] = field(default_factory=dict)  # 単軸要約 (後悔用)
    increment_vs_set: float = 0.0  # 試料群への追加情報 (§20.3)
    rank_range: Tuple[int, int] = (0, 0)
    dominated: bool = False


@dataclass
class SelectionResult:
    recommended: List[str]
    alternatives: List[str]
    undetermined_reasons: List[str]
    roles: Dict[str, str] = field(default_factory=dict)
    steps_log: List[str] = field(default_factory=list)


class ExperimentPlanner:
    """12手順の選択 (§20.5)。単一推奨ができない場合は代替案・判断不能理由を返す。"""

    def select(
        self,
        candidates: Mapping[str, UtilityVector],
        scenario_utils: Optional[Mapping[str, Mapping[str, float]]] = None,
        probabilistic: bool = False,
        costs: Optional[Mapping[str, float]] = None,
    ) -> SelectionResult:
        log: List[str] = []
        # 手順9: 明確に劣る候補を除く
        front = pareto_front(candidates)
        dominated = [i for i in candidates if i not in front]
        log.append(f"pareto front={front}, dominated={dominated}")
        if scenario_utils and not probabilistic:
            # 確率根拠なし: 順位安定性・最大後悔
            orders = {
                s: sorted(u, key=lambda c: -u[c]) for s, u in scenario_utils.items()
            }
            ranks = rank_stability(orders)
            regrets = minimax_regret(scenario_utils)
            log.append(f"rank_stability={ranks}")
            log.append(f"minimax_regret={regrets}")
            reversed_c = [c for c, (lo, hi) in ranks.items() if hi - lo > 0]
            if reversed_c:
                return SelectionResult(
                    recommended=[],
                    alternatives=front,
                    undetermined_reasons=[
                        f"シナリオにより順位が逆転する候補: {reversed_c}。"
                        "単一推奨を避け識別実験または代替案を返す (§20.4)",
                    ],
                    roles={c: ActionRole.DISCRIMINATION.value for c in front},
                    steps_log=log,
                )
            best = min(front, key=lambda c: regrets.get(c, float("inf")))
            return SelectionResult(
                recommended=[best],
                alternatives=[c for c in front if c != best],
                undetermined_reasons=[],
                roles={best: ActionRole.IMPROVEMENT.value},
                steps_log=log,
            )
        if probabilistic and scenario_utils:
            # 根拠ある確率: 期待値 (呼び出し側が根拠を保証)
            cands = {c for u in scenario_utils.values() for c in u}
            means = {c: sum(u.get(c, 0.0) for u in scenario_utils.values())
                     / max(len(scenario_utils), 1) for c in cands}
            best = max([c for c in front], key=lambda c: means.get(c, 0.0))
            log.append(f"expected_utility={means} (確率根拠ありの場合のみ有効)")
            return SelectionResult(
                recommended=[best],
                alternatives=[c for c in front if c != best],
                undetermined_reasons=[],
                roles={best: ActionRole.IMPROVEMENT.value},
                steps_log=log,
            )
        # シナリオ情報なし: 単独では決めない
        return SelectionResult(
            recommended=[],
            alternatives=front,
            undetermined_reasons=["不確かさ評価なしに単一推奨しない (§2.1-6, §20.4)"],
            roles={c: ActionRole.EXPLORATION.value for c in front},
            steps_log=log,
        )
