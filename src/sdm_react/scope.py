"""ScopeAudit: 支配機構とデータ適用範囲の監査 (設計書 §7).

計算前にどの物理を無視でき、どの物理を残すべきかを決める。
固相拡散を無条件の既定値にしない (§7.1)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Mechanism(str, Enum):
    """監査対象機構 (§7.2)。"""

    SOLID_DIFFUSION = "solid_diffusion"
    GB_TRANSPORT = "gb_transport"  # 粒界などの優先輸送
    INTERFACE_REACTION = "interface_reaction"
    OXIDE_FILM = "oxide_film"
    TRAPPING_PHASE = "trapping_phase"  # 中間相による捕捉
    BLOCKING_LAYER = "blocking_layer"  # 反応層による遮断
    VOID_DEBOND = "void_debond"  # 空隙・剥離
    LIQUID_FORMATION = "liquid_formation"
    DISSOLUTION_WETTING = "dissolution_wetting"  # 溶解・濡れ・浸透
    VOLUME_CHANGE = "volume_change"
    COOLING_TRANSFORM = "cooling_transform"


class MechanismStatus(str, Enum):
    """判定状態 (§7.3)。"""

    SUPPORTED = "supported"  # 関与を支持する観測がある
    BOUNDED_SMALL = "bounded_small"  # 目的観測量への影響が小さいと評価
    UNRESOLVED = "unresolved"  # 関与の可能性が未解決
    OUTSIDE_MODEL = "outside_model"  # 関与し得るが現在のモデルでは扱えない


@dataclass
class MechanismRecord:
    mechanism: Mechanism
    status: MechanismStatus = MechanismStatus.UNRESOLVED
    evidence: List[str] = field(default_factory=list)
    impact_evaluation: Optional[str] = None


class ScopeAudit:
    """支配機構の監査台帳。"""

    def __init__(self) -> None:
        self._records: Dict[Mechanism, MechanismRecord] = {
            m: MechanismRecord(mechanism=m) for m in Mechanism
        }

    def observe(self, mechanism: Mechanism, note: str) -> None:
        """関与を示唆する観測を記録し supported とする。"""
        rec = self._records[mechanism]
        rec.evidence.append(note)
        rec.status = MechanismStatus.SUPPORTED

    def bound_small(self, mechanism: Mechanism, evaluation: str) -> None:
        """目的観測量への影響が小さいことの評価を記録する。

        「観測されない」だけでは bounded_small にしない (§7.3)。
        定量的な影響評価の記述を必須とする。
        """
        if not evaluation or len(evaluation.strip()) < 10:
            raise ValueError(
                "bounded_small には定量的な影響評価の記述が必要。"
                "「観測されない」を自動変換してはならない (§7.3)。"
            )
        rec = self._records[mechanism]
        rec.impact_evaluation = evaluation
        rec.status = MechanismStatus.BOUNDED_SMALL

    def mark_outside_model(self, mechanism: Mechanism, note: str) -> None:
        rec = self._records[mechanism]
        rec.evidence.append(note)
        rec.status = MechanismStatus.OUTSIDE_MODEL

    def status_of(self, mechanism: Mechanism) -> MechanismStatus:
        return self._records[mechanism].status

    def unresolved(self) -> List[Mechanism]:
        return [
            m for m, r in self._records.items()
            if r.status == MechanismStatus.UNRESOLVED
        ]

    def outside_model(self) -> List[Mechanism]:
        return [
            m for m, r in self._records.items()
            if r.status == MechanismStatus.OUTSIDE_MODEL
        ]

    def branch(self, rank_sensitive: Optional[List[Mechanism]] = None) -> Dict[str, object]:
        """モデル分岐の判断 (§7.4)。

        重要な機構が未解決で、その有無により条件順位が変わる場合は
        (1) 複数シナリオ計算 (2) 適用範囲縮小 (3) 機構識別実験 のいずれかを選ぶ。
        """
        rank_sensitive = rank_sensitive or []
        critical = [m for m in rank_sensitive if self.status_of(m) == MechanismStatus.UNRESOLVED]
        outside = self.outside_model()
        if critical:
            return {
                "decision": "multi_scenario_or_discrimination",
                "critical_unresolved": [m.value for m in critical],
                "options": [
                    "複数機構のシナリオを計算する",
                    "適用範囲を狭める",
                    "機構識別実験を優先する",
                ],
            }
        if outside:
            return {
                "decision": "restrict_claims",
                "outside_model": [m.value for m in outside],
                "options": ["現モデルの主張を停止し適用範囲を明示する"],
            }
        return {"decision": "proceed", "options": ["現行モデルで進行可能"]}

    def report(self) -> Dict[str, Dict[str, object]]:
        return {
            m.value: {
                "status": r.status.value,
                "evidence": list(r.evidence),
                "impact_evaluation": r.impact_evaluation,
            }
            for m, r in self._records.items()
        }
