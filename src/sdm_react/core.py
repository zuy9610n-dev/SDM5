"""共通基盤: 用語・数量・状態変数・共通インターフェース (設計書 §4, §22.2-22.3).

このモジュールは物理モデルを含まない。単位・保存量・状態表現・結果状態の
共通規約だけを定める。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# §4.2-4.3 対象成分
# ---------------------------------------------------------------------------

ELEMENTS = ("Al", "Si", "Ti", "Fe", "Cu")
"""主五元素 (§4.3)。順序は表示・配列の既定順序としてのみ用いる。"""

OPTIONAL_COMPONENTS = ("O",)
"""追加追跡成分の例 (酸素など)。追跡時は主五元素基準と全成分基準を分ける (§4.3)。"""

ALL_COMPONENTS = ELEMENTS + OPTIONAL_COMPONENTS

# 原子量 [g/mol] (IUPAC 2021 略値)。換算専用であり材料固有の物性値ではない。
ATOMIC_MASS_G_MOL: Dict[str, float] = {
    "Al": 26.9815,
    "Si": 28.0855,
    "Ti": 47.867,
    "Fe": 55.845,
    "Cu": 63.546,
    "O": 15.999,
}

# 純元素の室温基準密度 [kg/m^3]。混合物体積の加成近似に使う場合のみ使用し、
# その旨を記録すること (N05)。O はバルク密度を持たないため除外。
DENSITY_KG_M3: Dict[str, float] = {
    "Al": 2700.0,
    "Si": 2330.0,
    "Ti": 4506.0,
    "Fe": 7874.0,
    "Cu": 8960.0,
}

GAS_CONSTANT_J_MOL_K = 8.314462618


# ---------------------------------------------------------------------------
# §5.3 三値判定と §9.7 到達可能性判定区分
# ---------------------------------------------------------------------------

Verdict3 = Literal["pass", "fail", "unknown"]
"""合格 / 不合格 / 判定不能 (§5.3)。判定不能を自動変換してはならない。"""


def combine_verdicts(verdicts: Sequence[Verdict3]) -> Verdict3:
    """必須条件群の三値結合 (§5.3)。

    - 全必須条件が合格 → pass
    - 一つ以上の必須条件が不合格 → fail
    - 不合格はないが未評価が残る → unknown
    """
    vs = list(verdicts)
    if any(v == "fail" for v in vs):
        return "fail"
    if all(v == "pass" for v in vs):
        return "pass"
    return "unknown"


class ReachabilityVerdict(str, Enum):
    """到達可能性の判定区分 (§9.7)。"""

    EXCLUDED_UNDER_ASSUMPTIONS = "excluded_under_assumptions"
    UNRESOLVED = "unresolved"
    MODEL_WITNESS = "model_witness"
    EXPERIMENTALLY_SUPPORTED = "experimentally_supported"


# ---------------------------------------------------------------------------
# §22.3 結果状態 (4 軸)
# ---------------------------------------------------------------------------


class ExecStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    INSUFFICIENT_INPUT = "insufficient_input"
    NUMERICAL_FAILURE = "numerical_failure"


class DomainStatus(str, Enum):
    VALIDATED = "validated"
    CALIBRATED_ONLY = "calibrated_only"
    EXTRAPOLATED = "extrapolated"
    UNSUPPORTED = "unsupported"


class IdentStatus(str, Enum):
    IDENTIFIABLE = "identifiable"
    PARTIALLY_IDENTIFIABLE = "partially_identifiable"
    NOT_IDENTIFIABLE = "not_identifiable"
    NOT_ASSESSED = "not_assessed"


class OutputNature(str, Enum):
    PHYSICAL_PREDICTION = "physical_prediction"
    EMPIRICAL_PREDICTION = "empirical_prediction"
    SCENARIO_RESULT = "scenario_result"
    OUTER_BOUND = "outer_bound"
    OBSERVATION_ONLY = "observation_only"


@dataclass
class ResultStatus:
    """結果の4軸状態。「計算成功」と「科学的に支持された予測」を分ける (§22.3)。"""

    execution: ExecStatus = ExecStatus.SUCCESS
    domain: DomainStatus = DomainStatus.UNSUPPORTED
    identifiability: IdentStatus = IdentStatus.NOT_ASSESSED
    nature: OutputNature = OutputNature.SCENARIO_RESULT
    notes: List[str] = field(default_factory=list)

    def is_scientifically_supported(self) -> bool:
        return self.execution == ExecStatus.SUCCESS and self.domain in (
            DomainStatus.VALIDATED,
            DomainStatus.CALIBRATED_ONLY,
        )


@dataclass
class Provenance:
    """来歴 (§23.2)。派生データは上流参照を持つ。"""

    code_version: str = "0.1.0"
    data_version: Optional[str] = None
    seed: Optional[int] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    upstream_refs: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# §22.2 共通インターフェース
# ---------------------------------------------------------------------------


@dataclass
class Capability:
    """モデルが宣言すべき能力・適用範囲 (§22.2)。"""

    name: str
    required_inputs: List[str] = field(default_factory=list)
    state_variables: List[str] = field(default_factory=list)
    conserved_quantities: List[str] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)
    boundary_conditions: List[str] = field(default_factory=list)
    predictable: List[str] = field(default_factory=list)
    unpredictable: List[str] = field(default_factory=list)
    domain: List[str] = field(default_factory=list)
    uncertainty_form: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    validation_records: List[str] = field(default_factory=list)


@dataclass
class ModelOutput:
    value: Any
    status: ResultStatus = field(default_factory=ResultStatus)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)


class BaseModel(ABC):
    """全物理・経験モデルの共通インターフェース (§22.2)。"""

    @abstractmethod
    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        """不足入力の名称リストを返す。空なら実行可能。"""

    @abstractmethod
    def describe_capabilities(self) -> Capability:
        """能力・保存量・単位・予測可能/不可能量を宣言する。"""

    @abstractmethod
    def describe_domain(self) -> Dict[str, Any]:
        """適用範囲 (温度・時間・形状・組成など) を宣言する。"""

    @abstractmethod
    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        """予測を実行する。"""

    def diagnose(self, result: ModelOutput) -> Dict[str, Any]:
        """数値診断 (保存誤差・非負性・収束など)。既定は診断辞書の返却。"""
        return dict(result.diagnostics)


# ---------------------------------------------------------------------------
# §4.1 材料状態 (概念的表現 S)
# ---------------------------------------------------------------------------


@dataclass
class MaterialState:
    """材料状態 S = {G, n, P, f, I, T, H} (§4.1)。

    全モデルがすべてを予測する必要はない。未予測・未測定は None
    (欠測) または候補リストとして保持する。
    """

    geometry: Optional[Dict[str, Any]] = None  # G: 領域形状と接続構造
    inventory_mol: Optional[Dict[str, Dict[str, float]]] = None  # n_{i,a}
    phases: Optional[Dict[str, List[str]]] = None  # P_a: 領域内の相/相候補
    phase_fractions: Optional[Dict[str, Dict[str, float]]] = None  # f_{p,a}
    interfaces: Optional[Dict[str, Any]] = None  # I: 界面状態
    temperature_K: Optional[float] = None  # T
    history: Optional[Dict[str, Any]] = None  # H
    provenance: Provenance = field(default_factory=Provenance)

    def missing(self) -> List[str]:
        out = []
        for name in (
            "geometry",
            "inventory_mol",
            "phases",
            "phase_fractions",
            "interfaces",
            "temperature_K",
            "history",
        ):
            if getattr(self, name) is None:
                out.append(name)
        return out


# ---------------------------------------------------------------------------
# §4.2 基本数量の換算 (N05)
# ---------------------------------------------------------------------------


def mole_fractions(n_mol: Mapping[str, float]) -> Dict[str, float]:
    """原子分率 x_{i,a} = n_i / Σn (§4.2)。"""
    total = float(sum(n_mol.values()))
    if total <= 0.0:
        raise ValueError("total moles must be positive")
    return {k: float(v) / total for k, v in n_mol.items()}


def mean_composition(
    inventories: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """複数領域の平均組成 x̄_i = Σ_a n / ΣΣn (§4.2)。

    異なる相の原子分率を無条件に体積平均しないこと。
    本関数は元素量基準の平均のみ行う。
    """
    totals: Dict[str, float] = {}
    for inv in inventories:
        for k, v in inv.items():
            totals[k] = totals.get(k, 0.0) + float(v)
    return mole_fractions(totals)


def mol_to_mass_kg(n_mol: float, element: str) -> float:
    return float(n_mol) * ATOMIC_MASS_G_MOL[element] / 1000.0


def mass_kg_to_mol(mass_kg: float, element: str) -> float:
    return float(mass_kg) * 1000.0 / ATOMIC_MASS_G_MOL[element]


def pure_molar_volume_m3_mol(element: str) -> float:
    """純元素のモル体積 [m^3/mol] (室温基準密度から)。"""
    mass_kg_mol = ATOMIC_MASS_G_MOL[element] / 1000.0
    return mass_kg_mol / DENSITY_KG_M3[element]


def mixture_volume_additive_m3(n_mol: Mapping[str, float]) -> float:
    """加成的純元素体積による混合物体積の近似 [m^3]。

    固溶・化合物形成による体積変化は含まない。使用時は仮定を記録すること。
    """
    return sum(
        float(n) * pure_molar_volume_m3_mol(el) for el, n in n_mol.items()
    )


def mass_fractions_to_mole_fractions(
    w: Mapping[str, float],
) -> Dict[str, float]:
    moles = {el: float(frac) / ATOMIC_MASS_G_MOL[el] for el, frac in w.items()}
    return mole_fractions(moles)


def mole_fractions_to_mass_fractions(
    x: Mapping[str, float],
) -> Dict[str, float]:
    masses = {
        el: float(frac) * ATOMIC_MASS_G_MOL[el] for el, frac in x.items()
    }
    total = sum(masses.values())
    if total <= 0.0:
        raise ValueError("total mass must be positive")
    return {el: m / total for el, m in masses.items()}


# ---------------------------------------------------------------------------
# §4.3 主五元素基準と全成分基準の分離
# ---------------------------------------------------------------------------


@dataclass
class CompositionRecord:
    """組成の記録 (§4.3)。再規格化値だけを保存してはならない。"""

    raw_values: Dict[str, float]  # 元の分析値 (分率または濃度)
    raw_total: float  # 分析合計
    quantification: Dict[str, Verdict3]  # 元素ごとの定量状態
    renormalized_5el: Optional[Dict[str, float]] = None  # 主五元素再規格化値
    extra_measured: Dict[str, bool] = field(default_factory=dict)  # 追加成分の測定状態

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, float],
        quantification: Optional[Mapping[str, Verdict3]] = None,
    ) -> "CompositionRecord":
        raw_d = {k: float(v) for k, v in raw.items()}
        total = float(sum(raw_d.values()))
        q: Dict[str, Verdict3] = (
            dict(quantification) if quantification else {k: "pass" for k in raw_d}
        )
        main = {el: raw_d[el] for el in ELEMENTS if el in raw_d}
        renorm = None
        if main and sum(main.values()) > 0:
            s = sum(main.values())
            renorm = {el: v / s for el, v in main.items()}
        extra = {el: (el in raw_d) for el in OPTIONAL_COMPONENTS}
        return cls(
            raw_values=raw_d,
            raw_total=total,
            quantification=q,
            renormalized_5el=renorm,
            extra_measured=extra,
        )
