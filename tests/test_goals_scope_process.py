"""§5 目標・§7 監査・§8 工程の試験。"""
import pytest

from sdm_react.goals import (
    Criterion,
    GoalCompiler,
    InformationType,
    TargetSpec,
)
from sdm_react.process import (
    Arrangement,
    FabDesign,
    InitialStateCandidate,
    InitialStateEnsemble,
    PredictionKind,
    ProcessState,
)
from sdm_react.scope import Mechanism, MechanismStatus, ScopeAudit


def _target():
    return TargetSpec(
        id="T1",
        information_type=InformationType.Q1,
        scientific_question="五元素反応領域は形成されるか",
        success_criteria=[
            Criterion("c1", "五元素定量", "n_elements_quantified", 5.0, ">="),
            Criterion("c2", "領域寸法", "region_um", 5.0, ">="),
        ],
        failure_criteria=[
            Criterion("f1", "全元素枯渇", "min_inventory_mol", 0.0, "<="),
        ],
    )


def test_goal_judge_three_valued():
    g = GoalCompiler()
    t = _target()
    assert g.judge(t, {"n_elements_quantified": 5.0, "region_um": 10.0}).overall == "pass"
    assert g.judge(t, {"n_elements_quantified": 3.0, "region_um": 10.0}).overall == "fail"
    j = g.judge(t, {"n_elements_quantified": 5.0})  # region 未測定
    assert j.overall == "unknown" and j.unknown == ["c2"]
    jf = g.judge(t, {"min_inventory_mol": 0.0})
    assert jf.overall == "fail"  # 失敗基準の成立


def test_goal_yaml_roundtrip():
    t = _target()
    d = t.to_dict()
    t2 = TargetSpec.from_dict(d)
    assert t2.id == "T1" and t2.information_type == InformationType.Q1
    assert len(t2.success_criteria) == 2


def test_scope_no_auto_bounded_small():
    audit = ScopeAudit()
    assert audit.status_of(Mechanism.SOLID_DIFFUSION) == MechanismStatus.UNRESOLVED
    with pytest.raises(ValueError):
        audit.bound_small(Mechanism.LIQUID_FORMATION, "短い")  # 記述不足は拒否
    audit.bound_small(Mechanism.LIQUID_FORMATION,
                      "示差熱分析で発熱なし、目的層厚への寄与<0.1umと評価")
    assert audit.status_of(Mechanism.LIQUID_FORMATION) == MechanismStatus.BOUNDED_SMALL
    audit.observe(Mechanism.OXIDE_FILM, "表面に酸化物を確認")
    assert audit.status_of(Mechanism.OXIDE_FILM) == MechanismStatus.SUPPORTED
    br = audit.branch(rank_sensitive=[Mechanism.GB_TRANSPORT])
    assert br["decision"] == "multi_scenario_or_discrimination"  # §7.4


def test_process_distribution_vs_set_not_confused():
    ps = ProcessState()
    design = FabDesign(id="D1", total_composition_mol={"Al": 1.0},
                       particle_size_um={"Al": 10.0},
                       arrangement=Arrangement.R, operable_vars=["arrangement"])
    cands = [InitialStateCandidate(id="c1"), InitialStateCandidate(id="c2")]
    ens = ps.predict_initial(design, cands)
    assert ens.kind == PredictionKind.PRE_FAB
    assert ens.is_distribution is False and ens.weights is None
    with pytest.raises(ValueError):
        InitialStateEnsemble(design.id, PredictionKind.PRE_FAB, cands,
                             is_distribution=False, weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        InitialStateEnsemble(design.id, PredictionKind.PRE_FAB, cands,
                             is_distribution=True, weights=[0.5])  # 数不一致


def test_process_post_fab_conditioning_is_separate():
    ps = ProcessState()
    design = FabDesign(id="D1", total_composition_mol={"Al": 1.0},
                       particle_size_um={"Al": 10.0})
    ens = ps.predict_initial(design, [InitialStateCandidate(id="c1"),
                                      InitialStateCandidate(id="c2")])
    cond = ps.condition_on_measurement(ens, "meas-1", ["c2"])
    assert cond.kind == PredictionKind.POST_FAB_CONDITIONAL  # §8.5
    assert [c.id for c in cond.candidates] == ["c2"]
    cov = ProcessState.check_coverage(cond.candidates[0])
    assert cov["connectivity_3d"] is False  # 2D から一意決定しない (§8.4)
