"""§14 冷却・観測と §15 対立説明の試験 (§25.3)。"""
import pytest

from sdm_react.alternatives import (
    ConsistencySpec,
    RivalHypothesis,
    RivalPrediction,
    consistent_hypotheses,
    discriminate,
    interval_separation,
    mixture_prediction,
)
from sdm_react.observation import (
    CoolingClass,
    CoolingHistory,
    MeasurementCondition,
    ObservabilityReport,
    ObservationModel,
    apply_cooling,
)


def _cond():
    return MeasurementCondition(
        id="A1", probe_volume_um3=1.0,
        detection_limit={"Al": 0.01, "Cu": 0.01},
        systematic_bias={"Al": 0.0, "Cu": 0.0},
        noise_sigma={"Al": 0.0, "Cu": 0.0},
    )


def test_observation_equation_and_dl():
    m = ObservationModel(_cond())
    out = m.predict({"state_composition": {"Al": 0.5, "Cu": 0.005},
                     "position_um": [1.0, 2.0], "seed": 1})
    pt = out.value["point"]
    assert pt.quantification["Al"] == "pass"
    assert pt.quantification["Cu"] == "unknown"  # 定量限界以下
    assert out.diagnostics["below_DL"] == ["Cu"]


def test_c4_restricts_high_T_claims():
    st = apply_cooling({"phase": "X"}, CoolingHistory(cooling_class=CoolingClass.C4))
    assert "high_T_claim_restriction" in st  # 室温解析は可・高温主張は制限 (§14.1)


def test_c1_requires_evidence():
    with pytest.raises(ValueError):
        apply_cooling({"x": 1}, CoolingHistory(cooling_class=CoolingClass.C1))
    ok = apply_cooling({"x": 1}, CoolingHistory(
        cooling_class=CoolingClass.C1, evidence="繰返し冷却試験で変化<検出限界"))
    assert ok["cooling_class"] == "C1"


def test_smoothing_only_when_validated():
    m = ObservationModel(_cond(), smoothing_validated=False)
    with pytest.raises(PermissionError):
        m.smoothed([{"Al": 1.0}, {"Al": 0.0}], [0.5, 0.5])
    m2 = ObservationModel(_cond(), smoothing_validated=True)
    assert m2.smoothed([{"Al": 1.0}, {"Al": 0.0}], [0.5, 0.5]) == {"Al": 0.5}


def test_observability_combines_axes():
    rep = ObservabilityReport(detectable="pass", bias_small="pass",
                              contamination_small="pass", depth_mixing_small="pass",
                              uncertainty_ok="pass",
                              structure_correspondence="unknown", cost_ok="pass")
    assert rep.overall() == "unknown"  # 固定最小層厚だけでは決めない (§14.4)


def test_false_intermediate_from_mixture():
    # 既知二相混合が偽中間組成を生む (§25.3)
    pred = mixture_prediction([{"Al": 1.0, "Cu": 0.0}, {"Al": 0.0, "Cu": 1.0}],
                              [0.5, 0.5])
    assert pred.predicted_composition == pytest.approx({"Al": 0.5, "Cu": 0.5})
    with pytest.raises(ValueError):
        mixture_prediction([{"Al": 1.0}], [0.3])


def test_consistency_set_requires_preregistered_tau():
    spec = ConsistencySpec("max_abs_deviation", 0.05,
                           registered_before_observation=False)
    rivals = [RivalPrediction(RivalHypothesis.UNIFORM_SINGLE_PHASE, {"Al": 0.5})]
    with pytest.raises(PermissionError):
        consistent_hypotheses({"Al": 0.5}, rivals, spec)  # 観測後の変更禁止 (§15.2)
    spec.registered_before_observation = True
    assert consistent_hypotheses({"Al": 0.52}, rivals, spec) == [
        RivalHypothesis.UNIFORM_SINGLE_PHASE]
    assert consistent_hypotheses({"Al": 0.9}, rivals, spec) == []


def test_single_vs_mixture_identification_limit():
    # 単相と混合が同じ観測に整合 → 一意認定しない (§25.3)
    spec = ConsistencySpec("max_abs_deviation", 0.02,
                           registered_before_observation=True)
    rivals = [
        RivalPrediction(RivalHypothesis.UNIFORM_SINGLE_PHASE, {"Al": 0.5, "Cu": 0.5}),
        mixture_prediction([{"Al": 1.0, "Cu": 0.0}, {"Al": 0.0, "Cu": 1.0}], [0.5, 0.5]),
    ]
    hits = consistent_hypotheses({"Al": 0.5, "Cu": 0.5}, rivals, spec)
    assert len(hits) == 2


def test_discrimination_separation_and_nonuniqueness():
    assert interval_separation((0.0, 0.1), (0.5, 0.6), 0.01) > 0
    assert interval_separation((0.0, 0.5), (0.4, 0.9), 0.01) <= 0
    rep = discriminate("a1", {"h1": {"Al": (0.0, 0.5)}, "h2": {"Al": (0.4, 0.9)}},
                       "Al", 0.01, cost=3.0)
    assert rep.uniquely_supported is None  # 一意に支持できない (§15.4)
    assert rep.cost == 3.0
    rep2 = discriminate("a2", {"h1": {"Al": (0.0, 0.1)}, "h2": {"Al": (0.5, 0.6)}},
                        "Al", 0.01)
    assert rep2.worst_case_separation > 0
