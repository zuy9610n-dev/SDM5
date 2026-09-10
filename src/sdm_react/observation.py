"""ObservationSolver: 冷却・仮想測定 (設計書 §14)。

形成状態と観測状態を分ける (§2.1)。冷却未評価の室温相を焼鈍温度の相として
扱わない (§2.2)。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from sdm_react.core import (
    BaseModel,
    Capability,
    DomainStatus,
    ExecStatus,
    IdentStatus,
    ModelOutput,
    OutputNature,
    ResultStatus,
    Verdict3,
)


class CoolingClass(str, Enum):
    """冷却の状態区分 (§14.1)。"""

    C1 = "C1"  # 目的観測量への影響が小さいと確認
    C2 = "C2"  # 冷却変化を物理モデルで扱う
    C3 = "C3"  # 冷却条件を含む経験モデルで予測
    C4 = "C4"  # 冷却影響が未評価


@dataclass
class CoolingHistory:
    """冷却履歴。"""

    time_s: List[float] = field(default_factory=list)
    temperature_K: List[float] = field(default_factory=list)
    cooling_class: CoolingClass = CoolingClass.C4
    evidence: str = ""  # C1 の場合は確認根拠が必須


@dataclass
class MeasurementCondition:
    """測定条件 A (§14.3)。"""

    id: str = ""
    probe_volume_um3: float = 0.0  # 分析体積
    depth_overlap: str = "unknown"  # 深さ方向の重なり評価
    detection_limit: Dict[str, float] = field(default_factory=dict)  # 元素別定量限界
    systematic_bias: Dict[str, float] = field(default_factory=dict)  # 系統誤差
    noise_sigma: Dict[str, float] = field(default_factory=dict)  # 偶然誤差
    registration: str = "unknown"  # 位置合わせ
    surface_state: str = "unknown"
    section_direction: str = "unknown"
    boundary_tilt: str = "unknown"
    point_selection: str = "unknown"  # 測定点の選択


@dataclass
class ObservedPoint:
    """仮想測定の1点。"""

    position_um: List[float]
    composition: Dict[str, float]  # 定量値 (原子分率)
    quantification: Dict[str, Verdict3]  # 元素別定量状態
    sigma: Dict[str, float] = field(default_factory=dict)


class ObservationModel(BaseModel):
    """観測式 y = H_A(S_RT) + ε (§14.2)。

    材料状態 → 信号発生 → 検出 → 定量処理の連鎖を明示する。
    単純な空間平滑化は有効近似として検証された範囲でのみ用いる。
    """

    def __init__(
        self,
        condition: MeasurementCondition,
        smoothing_validated: bool = False,
    ) -> None:
        self.cond = condition
        self.smoothing_validated = smoothing_validated

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        missing = []
        if "state_composition" not in inputs:
            missing.append("state_composition")
        return missing

    def describe_capabilities(self) -> Capability:
        return Capability(
            name="observation:H_A",
            required_inputs=["state_composition"],
            state_variables=["y"],
            conserved_quantities=[],
            units={"composition": "mole_fraction"},
            predictable=["仮想測定値", "定量状態"],
            unpredictable=["真の材料状態 (逆問題は別途)"],
            domain=["測定条件Aの校正範囲内"],
            uncertainty_form=["interval", "scenario"],
            diagnostics=["below_DL", "bias_applied"],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return {"measurement_condition": self.cond.id,
                "smoothing_validated": self.smoothing_validated}

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        missing = self.check_requirements(inputs)
        if missing:
            return ModelOutput(
                None,
                ResultStatus(execution=ExecStatus.INSUFFICIENT_INPUT,
                             notes=[f"missing: {missing}"]),
            )
        comp = dict(inputs["state_composition"])  # 真の局所組成
        pos = list(inputs.get("position_um", [0.0]))
        rng = inputs.get("rng") or random.Random(inputs.get("seed", 0))
        meas: Dict[str, float] = {}
        quant: Dict[str, Verdict3] = {}
        below_dl = []
        for el, x in comp.items():
            # 信号発生→検出→定量: 系統誤差 + 偶然誤差
            bias = self.cond.systematic_bias.get(el, 0.0)
            sigma = self.cond.noise_sigma.get(el, 0.0)
            y = x + bias + (rng.gauss(0.0, sigma) if sigma > 0 else 0.0)
            dl = self.cond.detection_limit.get(el, 0.0)
            if y < dl:
                quant[el] = "unknown"  # 定量限界以下は unknown として保持
                below_dl.append(el)
            else:
                quant[el] = "pass"
            meas[el] = max(y, 0.0)
        point = ObservedPoint(pos, meas, quant, dict(self.cond.noise_sigma))
        return ModelOutput(
            {"point": point},
            ResultStatus(
                domain=DomainStatus.CALIBRATED_ONLY,
                identifiability=IdentStatus.NOT_ASSESSED,
                nature=OutputNature.OBSERVATION_ONLY,
            ),
            {"below_DL": below_dl, "bias_applied": dict(self.cond.systematic_bias)},
        )

    def smoothed(
        self, compositions: List[Dict[str, float]], weights: List[float]
    ) -> Dict[str, float]:
        """分析体積内の混合の有効近似。検証範囲内でのみ使用 (§14.2)。"""
        if not self.smoothing_validated:
            raise PermissionError(
                "空間平滑化は有効近似として検証された範囲でのみ用いる (§14.2)。"
                "smoothing_validated=True の根拠が必要。"
            )
        out: Dict[str, float] = {}
        s = sum(weights)
        for comp, w in zip(compositions, weights):
            for el, x in comp.items():
                out[el] = out.get(el, 0.0) + x * w / s
        return out


@dataclass
class ObservabilityReport:
    """観測可能性の判定 (§14.4)。固定の最小層厚だけでは決めない。"""

    detectable: Verdict3 = "unknown"
    bias_small: Verdict3 = "unknown"
    contamination_small: Verdict3 = "unknown"
    depth_mixing_small: Verdict3 = "unknown"
    uncertainty_ok: Verdict3 = "unknown"
    structure_correspondence: Verdict3 = "unknown"
    cost_ok: Verdict3 = "unknown"
    notes: List[str] = field(default_factory=list)

    def overall(self) -> Verdict3:
        from sdm_react.core import combine_verdicts

        return combine_verdicts([
            self.detectable, self.bias_small, self.contamination_small,
            self.depth_mixing_small, self.uncertainty_ok,
            self.structure_correspondence, self.cost_ok,
        ])


def apply_cooling(
    state_at_hold: Mapping[str, Any],
    cooling: CoolingHistory,
) -> Dict[str, Any]:
    """冷却 C: 焼鈍終了状態 → 室温状態 (§13.1, §14.1)。

    C4 (未評価) でも室温観測の解析はできるが、高温相の存在や
    高温相平衡の主張には制限を付ける。
    """
    out = dict(state_at_hold)
    out["cooling_class"] = cooling.cooling_class.value
    if cooling.cooling_class == CoolingClass.C4:
        out["high_T_claim_restriction"] = (
            "冷却影響未評価のため高温相・高温平衡の主張は制限される (§14.1)"
        )
    elif cooling.cooling_class == CoolingClass.C1 and not cooling.evidence:
        raise ValueError("C1 には影響が小さいことの確認根拠が必要 (§14.1)")
    return out


def detection_sigma_from_counts(counts: float) -> float:
    """計数のポアソン統計からの目安 (例示)。"""
    return math.sqrt(max(counts, 0.0)) / max(counts, 1.0)
