"""§9 到達可能性の試験 (§25.2)。"""
import pytest

from sdm_react.core import ReachabilityVerdict
from sdm_react.reachability import (
    BoundBasis,
    CommonCapacity,
    TransportCap,
    assess_inventory,
    assess_transport,
    mark_experimentally_supported,
    mark_model_witness,
)


def test_feasible_case_not_excluded():
    rep = assess_inventory(
        [{"Al": 1.0, "Cu": 1.0}, {"Al": 0.5, "Cu": 2.0}],
        {"Al": 1.0, "Cu": 2.0},
        assumptions=["閉鎖系", "目標範囲の2点サンプル"],
    )
    # 実現可能な合成例を誤除外しない (§25.2)
    assert rep.verdict == ReachabilityVerdict.UNRESOLVED
    assert rep.nature == "outer_bound"


def test_shortage_detected_with_attachments():
    rep = assess_inventory(
        [{"Al": 5.0}],
        {"Al": 1.0},
        assumptions=["閉鎖系"],
        uncovered_mechanisms=["液相経路は未包含"],
    )
    assert rep.verdict == ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS
    assert rep.excluding_constraint is not None
    assert rep.overturn_conditions  # §9.7 必須添付
    rep.require_attachments_for_exclusion()


def test_most_favorable_in_range_used():
    # 目標範囲内の最も有利な状態で評価 (§9.3)
    rep = assess_inventory(
        [{"Al": 10.0}, {"Al": 0.1}], {"Al": 1.0}, assumptions=["閉鎖系"]
    )
    assert rep.verdict == ReachabilityVerdict.UNRESOLVED
    assert rep.details["min_req_mol"]["Al"] == pytest.approx(0.1)


def _caps(extra=True):
    caps = [
        TransportCap("s", "m", "Al", 3.0, BoundBasis.CONSERVATIVE, "保守的上限"),
        TransportCap("m", "t", "Al", 3.0, BoundBasis.CONSERVATIVE, "保守的上限"),
    ]
    if extra:
        caps.append(
            TransportCap("s", "t", "Al", 5.0, BoundBasis.CONSERVATIVE, "追加経路")
        )
    return caps


def test_transport_feasible_and_path_addition_monotone():
    # 輸送経路追加で外側到達集合が不合理に縮小しない (§25.2)
    base = assess_transport({"Al": 4.0}, _caps(extra=False),
                            target_region="t", source_regions=["s"],
                            assumptions=["緩和問題"])
    assert base.verdict == ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS
    added = assess_transport({"Al": 4.0}, _caps(extra=True),
                             target_region="t", source_regions=["s"],
                             assumptions=["緩和問題"])
    assert added.verdict == ReachabilityVerdict.UNRESOLVED  # 追加で充足へ


def test_transport_exclusion_reports_overturn():
    rep = assess_transport({"Al": 100.0}, _caps(extra=False),
                           target_region="t", source_regions=["s"],
                           assumptions=["緩和問題"])
    assert rep.verdict == ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS
    assert "transport" in (rep.excluding_constraint or "")
    assert rep.details["maxflow_mol"] == pytest.approx(3.0)


def test_common_capacity_constraint():
    cc = CommonCapacity("shared", [("s", "t", "Al"), ("s", "t", "Cu")],
                        5.0, BoundBasis.CONSERVATIVE)
    caps = [
        TransportCap("s", "t", "Al", 10.0, BoundBasis.CONSERVATIVE),
        TransportCap("s", "t", "Cu", 10.0, BoundBasis.CONSERVATIVE),
    ]
    rep = assess_transport({"Al": 4.0, "Cu": 4.0}, caps, [cc],
                           target_region="t", source_regions=["s"],
                           assumptions=["緩和問題"])
    assert rep.verdict == ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS


def test_no_strong_exclusion_without_basis():
    # 上限根拠なし → 強い不可能判定を返さない (§9.6, §25.2)
    caps = [TransportCap("s", "t", "Al", 0.001, BoundBasis.NONE)]
    rep = assess_transport({"Al": 100.0}, caps,
                           target_region="t", source_regions=["s"])
    assert rep.verdict == ReachabilityVerdict.UNRESOLVED


def test_relaxation_not_shown_as_guarantee():
    rep = assess_transport({"Al": 1.0}, _caps(extra=True),
                           target_region="t", source_regions=["s"])
    assert rep.nature == "outer_bound"  # 物理的実現保証として表示しない (§25.2)


def test_witness_and_experimental_marks():
    rep = assess_inventory([{"Al": 1.0}], {"Al": 2.0}, ["a"])
    mark_model_witness(rep, "run-1")
    assert rep.verdict == ReachabilityVerdict.MODEL_WITNESS
    mark_experimentally_supported(rep, "ev-1", scope="T=900K±10K")
    assert rep.verdict == ReachabilityVerdict.EXPERIMENTALLY_SUPPORTED
