"""§16・§17・§20・§23.3 計画選択の試験。"""
import pytest

from sdm_react.pipeline import default_ticket_for
from sdm_react.planner import (
    ActionKind,
    ActionRole,
    ExperimentPlanner,
    FabConstraints,
    UtilityVector,
    check_fabricable,
    dominates,
    minimax_regret,
    pareto_front,
    rank_stability,
    response_matrix,
)
from sdm_react.process import Arrangement


def test_pareto_removes_dominated():
    vecs = {
        "a": UtilityVector(3, 3, 3, 3, -1),
        "b": UtilityVector(1, 1, 1, 1, -5),  # a に優越される
        "c": UtilityVector(5, 0, 0, 0, -1),  # トレードオフで残る
    }
    assert dominates(vecs["a"], vecs["b"])
    assert not dominates(vecs["a"], vecs["c"])
    front = pareto_front(vecs)
    assert set(front) == {"a", "c"}


def test_rank_reversal_avoids_single_recommendation():
    plan = ExperimentPlanner()
    vecs = {"a": UtilityVector(1, 1, 1, 1, -1), "b": UtilityVector(1, 1, 1, 1, -1)}
    res = plan.select(vecs, {"s1": {"a": 10.0, "b": 0.0}, "s2": {"a": 0.0, "b": 10.0}},
                      probabilistic=False)
    assert res.recommended == []  # 順位逆転 → 単一推奨を避ける (§20.4)
    assert res.undetermined_reasons
    assert res.roles["a"] == ActionRole.DISCRIMINATION.value


def test_minimax_regret_and_stable_rank():
    reg = minimax_regret({"s1": {"a": 5.0, "b": 4.0}, "s2": {"a": 5.0, "b": 4.0}})
    assert reg["a"] == pytest.approx(0.0)
    assert reg["b"] == pytest.approx(1.0)
    ranks = rank_stability({"s1": ["a", "b"], "s2": ["a", "b"]})
    assert ranks["a"] == (0, 0)
    plan = ExperimentPlanner()
    vecs = {"a": UtilityVector(5, 0, 0, 1, -1), "b": UtilityVector(4, 0, 0, 1, -1)}
    res = plan.select(vecs, {"s1": {"a": 5.0, "b": 4.0}, "s2": {"a": 5.0, "b": 4.0}},
                      probabilistic=False)
    assert res.recommended == ["a"]


def test_no_scenario_no_single_pick():
    plan = ExperimentPlanner()
    res = plan.select({"a": UtilityVector(9, 9, 9, 9, -1)})
    assert res.recommended == []  # 不確かさ評価なしに単一推奨しない


def test_fabricability_with_error():
    cons = FabConstraints(arrangement_resolution_um=10.0, max_sample_mm=10.0)
    ok, _ = check_fabricable(Arrangement.P1, {"min_feature_um": 50.0, "size_mm": 5.0},
                             cons, fab_error_um=5.0)
    assert ok
    ok2, reasons = check_fabricable(Arrangement.P1, {"min_feature_um": 5.0},
                                    cons, fab_error_um=5.0)
    assert not ok2 and reasons  # 作製誤差で維持できない (§16.2)


def test_response_matrix_warns_coverage():
    rep = response_matrix(["x_Al"], ["u1"], [[2.0]], [0.1], [1.0])
    assert "warn" in rep and "被覆証明" in rep["warn"]  # §17.3


def test_action_ticket_preregistration():
    t = default_ticket_for("A-1", ["T1"], "exploration", "arrangement_change")
    assert t.kind == ActionKind.ARRANGEMENT_CHANGE
    assert t.preregistration_timestamp == ""
    t.preregister()
    assert t.preregistration_timestamp  # 実験前予測の固定 (§23.3)
    d = t.to_dict()["action"]
    assert d["role"] == "exploration" and "preregistration_timestamp" in d
