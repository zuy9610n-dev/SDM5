"""Reachability: 到達可能性の外側評価 (設計書 §9)。

詳細計算の前に、目標取得の必要条件を評価する。必要条件を満たすことは
実現の保証ではない (§9.1)。緩和問題の実現可能判定を物理的実現保証として
表示しない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from sdm_react.core import ReachabilityVerdict


class BoundBasis(str, Enum):
    """輸送上限の根拠 (§9.6)。"""

    VERIFIED_MODEL = "verified_model"
    CALIBRATED = "calibrated"
    CONSERVATIVE = "conservative"
    SCENARIO = "scenario"
    NONE = "none"  # 根拠なし: 有限値を捏造して不可能判定を行わない


@dataclass
class TransportCap:
    """経路ごとの輸送量上限 Q^{k,max} (§9.5)。"""

    frm: str
    to: str
    element: str
    qmax_mol: float
    basis: BoundBasis = BoundBasis.NONE
    note: str = ""


@dataclass
class CommonCapacity:
    """複数元素・相が共有する経路容量 (§9.5)。"""

    id: str
    members: List[Tuple[str, str, str]]  # (frm, to, element) の集合
    cap_mol: float
    basis: BoundBasis = BoundBasis.NONE


@dataclass
class ReachabilityReport:
    verdict: ReachabilityVerdict
    assumptions: List[str] = field(default_factory=list)
    excluding_constraint: Optional[str] = None
    uncovered_mechanisms: List[str] = field(default_factory=list)
    overturn_conditions: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)
    nature: str = "outer_bound"  # 常に外側評価として表示

    def require_attachments_for_exclusion(self) -> None:
        """不可能判定には必須添付 (§9.7)。"""
        if self.verdict == ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS:
            if not self.assumptions:
                raise ValueError("exclusion requires assumptions (§9.7)")
            if not self.excluding_constraint:
                raise ValueError("exclusion requires excluding constraint (§9.7)")
            if not self.overturn_conditions:
                raise ValueError("exclusion requires overturn conditions (§9.7)")


def assess_inventory(
    requirement_samples: Sequence[Dict[str, float]],
    available_max_mol: Dict[str, float],
    assumptions: Sequence[str],
    uncovered_mechanisms: Sequence[str] = (),
) -> ReachabilityReport:
    """原料量制約の必要条件評価 (§9.3)。

    目標組成・領域寸法が範囲指定の場合、許容範囲内の最も有利な状態
    (必要量最小) で評価する。範囲内最小でも不足なら除外できる。
    requirement_samples: 目標範囲を代表する要求量サンプル群。
    """
    if not requirement_samples:
        return ReachabilityReport(
            verdict=ReachabilityVerdict.UNRESOLVED,
            assumptions=list(assumptions),
            uncovered_mechanisms=list(uncovered_mechanisms),
            details={"reason": "no requirement samples given"},
        )
    elements = set(available_max_mol) | {
        el for s in requirement_samples for el in s
    }
    min_req = {
        el: min(float(s.get(el, 0.0)) for s in requirement_samples)
        for el in elements
    }
    shortage = {
        el: min_req[el] - float(available_max_mol.get(el, 0.0)) for el in elements
    }
    lacking = {el: v for el, v in shortage.items() if v > 0}
    if lacking:
        worst = max(lacking, key=lambda e: lacking[e])
        rep = ReachabilityReport(
            verdict=ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS,
            assumptions=list(assumptions),
            excluding_constraint=(
                f"inventory: min_req[{worst}]={min_req[worst]:.6g} mol > "
                f"avail_max={available_max_mol.get(worst, 0.0):.6g} mol"
            ),
            uncovered_mechanisms=list(uncovered_mechanisms),
            overturn_conditions=[
                "利用可能量の上限が増える場合",
                "目標範囲内に要求量がより小さい状態がある場合",
                "未包含の供給経路が存在する場合",
            ],
            details={"min_req_mol": min_req, "shortage_mol": lacking},
        )
        rep.require_attachments_for_exclusion()
        return rep
    return ReachabilityReport(
        verdict=ReachabilityVerdict.UNRESOLVED,
        assumptions=list(assumptions),
        uncovered_mechanisms=list(uncovered_mechanisms),
        details={
            "min_req_mol": min_req,
            "note": "必要条件を満たすが実現経路は未確認 (§9.7 unresolved)",
        },
    )


def _max_flow(
    edges: Dict[Tuple[str, str], float], source: str, sink: str
) -> float:
    """Edmonds-Karp による最大フロー (単一元素の外側評価用)。"""
    residual: Dict[Tuple[str, str], float] = dict(edges)
    nodes = {source, sink}
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)

    def neighbors(u: str) -> List[str]:
        out = []
        for v in nodes:
            if residual.get((u, v), 0.0) > 0:
                out.append(v)
        return out

    flow = 0.0
    while True:
        parent: Dict[str, Optional[str]] = {source: None}
        queue = [source]
        for u in queue:
            for v in neighbors(u):
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path_cap = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            assert u is not None
            path_cap = min(path_cap, residual.get((u, v), 0.0))
            v = u
        v = sink
        while v != source:
            u = parent[v]
            assert u is not None
            residual[(u, v)] = residual.get((u, v), 0.0) - path_cap
            residual[(v, u)] = residual.get((v, u), 0.0) + path_cap
            v = u
        flow += path_cap
    return flow


def assess_transport(
    demand_mol: Dict[str, float],
    caps: Sequence[TransportCap],
    common_caps: Sequence[CommonCapacity] = (),
    target_region: str = "",
    source_regions: Sequence[str] = (),
    assumptions: Sequence[str] = (),
    uncovered_mechanisms: Sequence[str] = (),
) -> ReachabilityReport:
    """輸送量制約の外側評価 (§9.5)。

    上り坂拡散・界面平衡・反応順序を十分に表現しないため、
    実現可能性ではなく供給面からの外側評価として扱う。
    根拠のない上限では強い不可能判定を返さない (§9.6)。
    """
    assumptions = list(assumptions) + [
        "緩和問題による外側評価。上り坂拡散・界面平衡・反応順序は未表現 (§9.5)",
    ]
    grounded = [c for c in caps if c.basis != BoundBasis.NONE]
    ungrounded = [c for c in caps if c.basis == BoundBasis.NONE]
    if ungrounded:
        return ReachabilityReport(
            verdict=ReachabilityVerdict.UNRESOLVED,
            assumptions=assumptions,
            uncovered_mechanisms=list(uncovered_mechanisms)
            + [f"上限根拠なし: {c.frm}->{c.to} {c.element}" for c in ungrounded],
            overturn_conditions=["輸送上限の根拠が得られた場合に再評価"],
            details={
                "reason": "上限の根拠がないため強い不可能判定を行わない (§9.6)",
                "demand_mol": dict(demand_mol),
            },
        )
    # 元素別の最大フロー評価
    for el, need in demand_mol.items():
        edges: Dict[Tuple[str, str], float] = {}
        for c in grounded:
            if c.element != el:
                continue
            key = (c.frm, c.to)
            edges[key] = edges.get(key, 0.0) + c.qmax_mol
        for src in source_regions:
            edges.setdefault((f"__S__", src), float("inf"))
        edges.setdefault((target_region, "__T__"), float("inf"))
        if target_region:
            capable = _max_flow(edges, "__S__", "__T__")
        else:
            capable = sum(edges.values())
        if capable < need:
            rep = ReachabilityReport(
                verdict=ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS,
                assumptions=assumptions,
                excluding_constraint=(
                    f"transport: element={el} demand={need:.6g} mol > "
                    f"maxflow={capable:.6g} mol"
                ),
                uncovered_mechanisms=list(uncovered_mechanisms),
                overturn_conditions=[
                    "輸送上限が増える場合 (温度・時間・経路の追加)",
                    "未包含の重要経路 (液相・粒界等) が存在する場合 (§9.6)",
                    "要求量が減る場合",
                ],
                details={"element": el, "demand_mol": need, "maxflow_mol": capable},
            )
            rep.require_attachments_for_exclusion()
            return rep
    # 共通容量制約の必要条件
    for cc in common_caps:
        used = 0.0
        for frm, to, el in cc.members:
            used += sum(
                c.qmax_mol
                for c in grounded
                if (c.frm, c.to, c.element) == (frm, to, el)
            )
        # 需要側の必要条件: 需要合計が共通容量を超えれば除外
        need_sum = sum(
            demand_mol.get(el, 0.0) for (_f, _t, el) in cc.members
        )
        if need_sum > cc.cap_mol:
            rep = ReachabilityReport(
                verdict=ReachabilityVerdict.EXCLUDED_UNDER_ASSUMPTIONS,
                assumptions=assumptions,
                excluding_constraint=(
                    f"common capacity {cc.id}: demand_sum={need_sum:.6g} > "
                    f"cap={cc.cap_mol:.6g}"
                ),
                uncovered_mechanisms=list(uncovered_mechanisms),
                overturn_conditions=["共通容量の拡大または需要の低減"],
                details={"id": cc.id, "used_bound": used},
            )
            rep.require_attachments_for_exclusion()
            return rep
    return ReachabilityReport(
        verdict=ReachabilityVerdict.UNRESOLVED,
        assumptions=assumptions,
        uncovered_mechanisms=list(uncovered_mechanisms),
        details={
            "demand_mol": dict(demand_mol),
            "note": "供給面の必要条件を満たす。実現保証ではない (§9.1)",
        },
    )


def mark_model_witness(report: ReachabilityReport, run_id: str) -> ReachabilityReport:
    """具体的なモデル計算で実現案が得られた場合 (§9.7 model_witness)。"""
    report.verdict = ReachabilityVerdict.MODEL_WITNESS
    report.details["witness_run"] = run_id
    return report


def mark_experimentally_supported(
    report: ReachabilityReport, evidence_id: str, scope: str
) -> ReachabilityReport:
    """指定範囲で実験確認がある場合 (§9.7)。"""
    report.verdict = ReachabilityVerdict.EXPERIMENTALLY_SUPPORTED
    report.details["evidence"] = evidence_id
    report.details["scope"] = scope
    return report
