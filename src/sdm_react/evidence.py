"""EvidenceStore: 材料情報と主張の証拠の保持 (設計書 §18, §19)。

- 同じ組成でも相・温度が異なれば別情報 (§18.1)
- 点密度解析は候補探索用。品質付き成果の水増し防止 (§18.2, §25.4)
- 評価測定と候補探索測定を分ける (§19.1)
- 画素数増加を独立試料数増加として扱わない (§2.2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sdm_react.core import Verdict3, combine_verdicts


class IndependenceLevel(str, Enum):
    """独立性の階層 (§19.3)。画素は最下層。"""

    LOT = "lot"  # 原料ロット
    FAB_BATCH = "fab_batch"  # 作製バッチ
    HEAT_BATCH = "heat_batch"  # 熱処理バッチ
    SAMPLE = "sample"  # 試料
    SECTION = "section"  # 断面
    FIELD = "field"  # 視野
    REGION = "region"  # 反応領域
    PIXEL = "pixel"  # 画素


@dataclass
class EvidenceUnit:
    """情報単位 (§18.1)。"""

    id: str
    sample_id: str
    position: str = ""
    temperature_K: Optional[float] = None
    thermal_history: str = ""
    local_mean_composition: Dict[str, float] = field(default_factory=dict)
    intra_phase_composition: Dict[str, float] = field(default_factory=dict)
    phase_or_candidates: List[str] = field(default_factory=list)
    coexistence_partition: str = ""
    microstructure_scale_um: Optional[float] = None
    method: str = ""
    quality: Verdict3 = "unknown"
    independence: IndependenceLevel = IndependenceLevel.PIXEL
    independence_id: str = ""  # 例: 試料ID・バッチID
    alternatives: List[str] = field(default_factory=list)
    supportable_claims: List[str] = field(default_factory=list)

    def info_key(self) -> Tuple[str, str, str]:
        """同じ組成でも相・温度が異なれば別情報 (§18.1)。"""
        comp = ",".join(f"{k}={v:.4g}" for k, v in sorted(self.local_mean_composition.items()))
        phase = "+".join(sorted(self.phase_or_candidates))
        return (comp, phase, str(self.temperature_K))


@dataclass
class CoexistenceRecord:
    """状態図情報の証拠構造。隣接と平衡共存を同一視しない (§18.3)。"""

    adjacency: Verdict3 = "unknown"  # 実空間での隣接
    coexistence_same_region: Verdict3 = "unknown"  # 同一局所領域での共存
    partition: Verdict3 = "unknown"  # 相間分配
    local_equilibrium_consistency: Verdict3 = "unknown"
    temporal_stability: Verdict3 = "unknown"
    cross_initial_reproduction: Verdict3 = "unknown"
    cooling_dependence: Verdict3 = "unknown"
    notes: List[str] = field(default_factory=list)


@dataclass
class DensityAnalysisSpec:
    """点密度解析の必須事項 (§18.2)。"""

    metric: str = "euclidean_composition"
    bandwidth: float = 0.0
    boundary_handling: str = ""
    zero_dl_handling: str = ""
    weights: str = "uniform"  # 抽出重み
    real_space_mapping: str = ""
    stratification: str = ""  # 相内部候補と混合候補の層別


def gaussian_kde_1d(
    samples: Sequence[float], points: Sequence[float], bandwidth: float
) -> List[float]:
    """1次元ガウス KDE (例示用。組成ノイズ試験 §25.4 に使用)。"""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    out = []
    n = len(samples)
    norm = n * bandwidth * math.sqrt(2.0 * math.pi)
    for p in points:
        s = sum(math.exp(-0.5 * ((p - x) / bandwidth) ** 2) for x in samples)
        out.append(s / norm)
    return out


class SamplingMode(str, Enum):
    EVALUATION = "evaluation"  # 評価測定: 共通枠・系統/確率抽出
    EXPLORATION = "exploration"  # 候補探索測定: 有望領域の深掘り


@dataclass
class AreaFraction:
    """面積割合 Y_q = A_q / A_W (§19.2)。合格・不合格・判定不能を併記。"""

    A_q_pass: float
    A_q_fail: float
    A_q_unknown: float
    A_W: float
    conversion_2d_to_3d: str = ""  # 変換時は仮定を明示

    def fractions(self) -> Dict[str, float]:
        if self.A_W <= 0:
            raise ValueError("A_W must be positive")
        return {
            "pass": self.A_q_pass / self.A_W,
            "fail": self.A_q_fail / self.A_W,
            "unknown": self.A_q_unknown / self.A_W,
        }


class EvidenceStore:
    """材料情報と主張の証拠の台帳 (§18, §19, §23.1 evidence/claim)。"""

    def __init__(self) -> None:
        self.units: Dict[str, EvidenceUnit] = {}
        self.coexistence: Dict[str, CoexistenceRecord] = {}
        self.sampling_modes: Dict[str, SamplingMode] = {}

    def add(self, unit: EvidenceUnit, sampling: SamplingMode) -> None:
        self.units[unit.id] = unit
        self.sampling_modes[unit.id] = sampling

    def count_independent(
        self, level: IndependenceLevel, quality: Optional[Verdict3] = None
    ) -> int:
        """指定階層での独立単位数を数える。画素≠独立試料 (§2.2, §19.3)。"""
        ids = set()
        for u in self.units.values():
            if quality is not None and u.quality != quality:
                continue
            # 指定階層以上の粒度でまとめる: 簡易実装として independence_id を使用
            if u.independence.value == level.value or _is_finer(u.independence, level):
                ids.add((level.value, u.independence_id or u.id))
        return len(ids)

    def overall_quality(self, ids: Sequence[str]) -> Verdict3:
        return combine_verdicts([self.units[i].quality for i in ids])

    def evaluation_only_ids(self) -> List[str]:
        """試料全体の頻度推定には評価測定のみ使う (§19.1)。"""
        return [i for i, m in self.sampling_modes.items()
                if m == SamplingMode.EVALUATION]


def _is_finer(a: IndependenceLevel, b: IndependenceLevel) -> bool:
    order = [l for l in IndependenceLevel]
    return order.index(a) > order.index(b)
