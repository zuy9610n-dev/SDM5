"""ProcessState: 作製条件と初期状態候補の接続 (設計書 §8).

作製前予測 p(S0|d_fab) または候補集合 S0(d_fab) と、
作製後条件付き予測を分離する (§8.5)。確率分布と無重み候補集合を混同しない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PredictionKind(str, Enum):
    PRE_FAB = "pre_fab"  # 作製前予測
    POST_FAB_CONDITIONAL = "post_fab_conditional"  # 作製後条件付き予測


class Arrangement(str, Enum):
    """候補配置の種類 (§16.1)。"""

    R = "R"  # 無作為混合
    P1 = "P1"  # 距離制御
    P2 = "P2"  # 接触順序制御
    P3 = "P3"  # 局在供給
    P4 = "P4"  # 多方向供給
    P5 = "P5"  # 複数反応モジュール


@dataclass
class FabDesign:
    """作製設計 d_fab (§8.1)。実験で変更できない変数は含めない。"""

    id: str
    total_composition_mol: Dict[str, float]  # 全体配合 [mol] (元素別)
    particle_size_um: Dict[str, float]  # 元素別代表粒径 [um]
    arrangement: Arrangement = Arrangement.R
    arrangement_detail: Dict[str, Any] = field(default_factory=dict)
    mixing: Dict[str, Any] = field(default_factory=dict)
    pressing: Dict[str, Any] = field(default_factory=dict)
    sintering_history: List[Dict[str, Any]] = field(default_factory=list)
    operable_vars: List[str] = field(default_factory=list)  # 操作変数の宣言

    def total_moles(self) -> float:
        return float(sum(self.total_composition_mol.values()))


# §8.4 初期に取得する構造情報の優先順位
STRUCTURE_PRIORITY = [
    "residual_inventory",  # 元素別の残存原料量
    "contact_types",  # 主要な接触類型
    "reaction_layers",  # 反応層・先行生成物
    "source_distances",  # 供給源間距離
    "voids",  # 空隙・剥離
    "contact_areas",  # 接触面積
    "connectivity_3d",  # 三次元連結性
]


@dataclass
class InitialStateCandidate:
    """焼結後初期状態の候補 (§8.2, §8.4)。"""

    id: str
    residual_inventory_mol: Dict[str, float] = field(default_factory=dict)
    contact_types: List[str] = field(default_factory=list)
    reaction_layers: List[Dict[str, Any]] = field(default_factory=list)
    source_distances_um: Dict[str, float] = field(default_factory=dict)
    voids: Dict[str, Any] = field(default_factory=dict)
    contact_areas_um2: Dict[str, float] = field(default_factory=dict)
    connectivity_3d: Optional[str] = None  # 2D像からは一意に決めない
    source: str = ""  # 取得方法 (§8.3): sister_sample / structure_model / ...
    notes: List[str] = field(default_factory=list)


@dataclass
class InitialStateEnsemble:
    """初期状態のアンサンブル。

    - 分布を正当化できる場合のみ is_distribution=True (p(S0|d_fab))
    - そうでなければ無重みの候補集合 (S0(d_fab)) として扱う (§8.2)
    """

    design_id: str
    kind: PredictionKind
    candidates: List[InitialStateCandidate] = field(default_factory=list)
    is_distribution: bool = False
    weights: Optional[List[float]] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.is_distribution:
            if self.weights is None or len(self.weights) != len(self.candidates):
                raise ValueError(
                    "分布形式では候補数と一致する重みが必要 (§8.2)。"
                )
            s = sum(self.weights)
            if s <= 0:
                raise ValueError("weights must sum to positive")
            self.weights = [w / s for w in self.weights]
        elif self.weights is not None:
            raise ValueError(
                "無重み候補集合に重みを付けてはならない。分布と混同しないこと (§8.2)。"
            )


class ProcessState:
    """作製条件→初期状態候補の写像 (§8)。"""

    def predict_initial(
        self,
        design: FabDesign,
        candidates: List[InitialStateCandidate],
        is_distribution: bool = False,
        weights: Optional[List[float]] = None,
    ) -> InitialStateEnsemble:
        """作製前予測: 設計条件から初期状態候補を構築する。

        候補の生成ロジック (工程モデル等) は呼び出し側が与える。
        本メソッドは形式 (分布/集合) の整合性と来歴区分を保証する。
        """
        return InitialStateEnsemble(
            design_id=design.id,
            kind=PredictionKind.PRE_FAB,
            candidates=candidates,
            is_distribution=is_distribution,
            weights=weights,
            notes=[f"design={design.id}", f"arrangement={design.arrangement.value}"],
        )

    def condition_on_measurement(
        self,
        ensemble: InitialStateEnsemble,
        measurement_id: str,
        selected_ids: List[str],
    ) -> InitialStateEnsemble:
        """作製後条件付き予測: 実測した焼結後状態で候補を絞る (§8.5)。

        断面観察が破壊的な場合、焼鈍前後の同一粒子対応を当然視しない
        (§8.3): 対応付けの根拠を notes に残すことを呼び出し側に求める。
        """
        selected = [c for c in ensemble.candidates if c.id in selected_ids]
        weights = None
        if ensemble.is_distribution and ensemble.weights is not None:
            idx = [i for i, c in enumerate(ensemble.candidates) if c.id in selected_ids]
            weights = [ensemble.weights[i] for i in idx]
            s = sum(weights)
            weights = [w / s for w in weights] if s > 0 else None
        return InitialStateEnsemble(
            design_id=ensemble.design_id,
            kind=PredictionKind.POST_FAB_CONDITIONAL,
            candidates=selected,
            is_distribution=ensemble.is_distribution,
            weights=weights,
            notes=[
                f"conditioned on measurement={measurement_id}",
                "破壊観察の場合は同一粒子対応の根拠を別途記録すること (§8.3)",
            ],
        )

    @staticmethod
    def check_coverage(
        candidate: InitialStateCandidate,
    ) -> Dict[str, bool]:
        """§8.4 の優先情報の取得状況を確認する。"""
        return {
            "residual_inventory": bool(candidate.residual_inventory_mol),
            "contact_types": bool(candidate.contact_types),
            "reaction_layers": True,  # 空=なし の可能性があるため常に True
            "source_distances": bool(candidate.source_distances_um),
            "voids": bool(candidate.voids),
            "contact_areas": bool(candidate.contact_areas_um2),
            "connectivity_3d": candidate.connectivity_3d is not None,
        }
