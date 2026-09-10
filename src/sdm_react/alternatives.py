"""AlternativeExplorer: 対立説明探索 (設計書 §15)。

現在の観測を説明できるが材料学的意味が異なる状態を探索し、
主張を一意にできない原因を明らかにする。最良適合の一点選びではない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class RivalHypothesis(str, Enum):
    """初期の対立説明ライブラリ (§15.3)。網羅を主張しない。"""

    UNIFORM_SINGLE_PHASE = "uniform_single_phase"
    INTRA_PHASE_GRADIENT = "intra_phase_gradient"
    FINE_MIXTURE_KNOWN = "fine_mixture_known"
    THIN_LAYER_AVERAGED = "thin_layer_averaged"
    DEPTH_OVERLAP = "depth_overlap"
    UNREACTED_FEED = "unreacted_feed"
    OXIDE_IMPURITY = "oxide_impurity"
    QUANT_ANOMALY = "quant_anomaly"
    MISREGISTRATION = "misregistration"
    COOLING_TRANSFORM = "cooling_transform"


@dataclass
class ConsistencySpec:
    """整合集合 S(y) = {S: D(y,H(S)) ≤ τ} の定義 (§15.2)。

    D と τ を観測後に都合よく変更しない: τ は観測前に登録する。
    """

    metric: str  # 例: "max_abs_deviation", "rss"
    tau: float
    registered_before_observation: bool = False
    notes: List[str] = field(default_factory=list)

    def check_usable(self) -> None:
        if not self.registered_before_observation:
            raise PermissionError(
                "D と τ は観測前に登録すること。観測後の都合よい変更は禁止 (§15.2)。"
            )


def deviation(
    observed: Mapping[str, float],
    predicted: Mapping[str, float],
    metric: str = "max_abs_deviation",
) -> float:
    els = set(observed) | set(predicted)
    diffs = [abs(observed.get(el, 0.0) - predicted.get(el, 0.0)) for el in els]
    if metric == "max_abs_deviation":
        return max(diffs) if diffs else 0.0
    if metric == "rss":
        return math.sqrt(sum(d * d for d in diffs))
    raise ValueError(f"unknown metric: {metric}")


@dataclass
class RivalPrediction:
    hypothesis: RivalHypothesis
    predicted_composition: Dict[str, float]
    interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


def mixture_prediction(
    endmembers: Sequence[Mapping[str, float]],
    fractions: Sequence[float],
    hypothesis: RivalHypothesis = RivalHypothesis.FINE_MIXTURE_KNOWN,
) -> RivalPrediction:
    """既知相の微細混合などの混合予測。偽中間組成を生みうる (§25.3)。"""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    out: Dict[str, float] = {}
    for em, f in zip(endmembers, fractions):
        for el, x in em.items():
            out[el] = out.get(el, 0.0) + x * f
    return RivalPrediction(hypothesis, out, params={"fractions": list(fractions)})


def consistent_hypotheses(
    observed: Mapping[str, float],
    rivals: Sequence[RivalPrediction],
    spec: ConsistencySpec,
) -> List[RivalHypothesis]:
    """観測に整合する対立説明の集合 S(y) (§15.2)。"""
    spec.check_usable()
    out = []
    for r in rivals:
        if deviation(observed, r.predicted_composition, spec.metric) <= spec.tau:
            out.append(r.hypothesis)
    return out


@dataclass
class DiscriminationReport:
    """識別実験の評価 (§15.4)。"""

    action: str
    separations: Dict[str, float] = field(default_factory=dict)  # 仮説対 -> 分離度
    worst_case_separation: float = 0.0
    inconclusive_range: str = ""
    cost: float = 0.0
    recommendation: str = ""
    uniquely_supported: Optional[str] = None


def interval_separation(
    a: Tuple[float, float], b: Tuple[float, float], measurement_error: float
) -> float:
    """測定誤差を考慮した予測区間の分離。>0 で分離。"""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    overlap = max(0.0, hi - lo)
    gap = 0.0 if overlap > 0 else min(abs(a[1] - b[0]), abs(b[1] - a[0]))
    return gap - overlap - 2.0 * measurement_error


def discriminate(
    action: str,
    predictions: Mapping[str, Dict[str, Tuple[float, float]]],
    metric_element: str,
    measurement_error: float,
    cost: float = 0.0,
    expected_info_gain: Optional[Dict[str, float]] = None,
) -> DiscriminationReport:
    """候補行動の識別性評価 (§15.4)。

    確率に根拠がある場合のみ期待情報利得を使用できる。
    根拠がなければ予測区間の分離・最悪条件・判定不能範囲・費用を比較する。
    """
    hyps = list(predictions)
    seps: Dict[str, float] = {}
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            a = predictions[hyps[i]][metric_element]
            b = predictions[hyps[j]][metric_element]
            seps[f"{hyps[i]} vs {hyps[j]}"] = interval_separation(a, b, measurement_error)
    worst = min(seps.values()) if seps else 0.0
    if worst > 0:
        rec = f"行動 {action} は全仮説対を分離する (最悪分離={worst:.3g})"
    else:
        rec = (
            f"行動 {action} では分離しない仮説対が残る (最悪分離={worst:.3g})。"
            "「一意に支持できない」と返す (§15.4)"
        )
    if expected_info_gain is not None:
        rec += f"。期待情報利得(確率根拠あり): {expected_info_gain}"
    return DiscriminationReport(
        action=action,
        separations=seps,
        worst_case_separation=worst,
        inconclusive_range=f"metric={metric_element}, err={measurement_error}",
        cost=cost,
        recommendation=rec,
        uniquely_supported=None if worst <= 0 else "separable(要追試)",
    )
