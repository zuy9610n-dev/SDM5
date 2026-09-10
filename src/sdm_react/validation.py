"""Validation: 数値検証・独立実験検証の管理 (設計書 §21, §25)。

検証段階 V1-V4 (§1.2):
- V1: 保存則・数値計算が正しい
- V2: 未使用条件の観測量を予測できる
- V3: 作製前に条件比較ができる
- V4: 推奨によって材料情報の取得効率が改善する
V2 の成功は V3・V4 の成立を意味しない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ValidationStage(str, Enum):
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"


class UncertaintyClass(str, Enum):
    """不確かさの分類 (§21.1)。一つのノイズへまとめない。"""

    FABRICATION = "fabrication"  # 作製変動
    INITIAL_STATE = "initial_state"  # 初期状態不確かさ
    MEASUREMENT = "measurement"  # 測定誤差
    PARAMETER = "parameter"  # パラメータ不確かさ
    MODEL_STRUCTURE = "model_structure"  # モデル構造不確かさ
    EXTRAPOLATION = "extrapolation"  # 適用範囲外への外挿
    NUMERICAL = "numerical"  # 数値誤差


@dataclass
class CalibrationPlan:
    """校正前に固定するもの (§21.2)。"""

    free_parameters: List[str] = field(default_factory=list)
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    used_observations: List[str] = field(default_factory=list)
    held_out_validation: List[str] = field(default_factory=list)
    initial_state_handling: str = ""
    boundary_conditions: str = ""
    error_model: str = ""
    parameter_range_basis: str = ""


@dataclass
class ValidationRecord:
    """検証記録 (§23.1 validation_record)。"""

    id: str
    stage: ValidationStage
    target: str
    held_out: List[str] = field(default_factory=list)
    metric: str = ""
    value: Optional[float] = None
    passed: Optional[bool] = None
    prediction_kind: str = ""  # pre_fab / post_fab_conditional を区別 (§8.5)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# §25.1 数値検証 N01-N12 の検査関数 (pytest からも呼ぶ)
# ---------------------------------------------------------------------------


def check_closed_conservation(
    initial: Mapping[str, float],
    final: Mapping[str, float],
    tol_mol: float,
) -> Dict[str, float]:
    """N01: 閉鎖系元素保存。"""
    res = {el: final.get(el, 0.0) - initial.get(el, 0.0) for el in initial}
    for el, r in res.items():
        if abs(r) > tol_mol:
            raise AssertionError(f"N01 violated for {el}: {r:.6g} > {tol_mol:.6g}")
    return res


def check_open_balance(
    initial: Mapping[str, float],
    final: Mapping[str, float],
    net_external_mol: Mapping[str, float],
    tol_mol: float,
) -> Dict[str, float]:
    """N02: 外部流束を含む収支。"""
    res = {
        el: final.get(el, 0.0) - initial.get(el, 0.0) - net_external_mol.get(el, 0.0)
        for el in initial
    }
    for el, r in res.items():
        if abs(r) > tol_mol:
            raise AssertionError(f"N02 violated for {el}: {r:.6g}")
    return res


def check_antisymmetry(
    flow_ab: float, flow_ba: float, tol_mol_s: float
) -> float:
    """N03: 領域間流量の一致 Φ_ab = -Φ_ba。"""
    res = flow_ab + flow_ba
    if abs(res) > tol_mol_s:
        raise AssertionError(f"N03 violated: {res:.6g}")
    return res


def check_nonnegative(values: Mapping[str, float]) -> None:
    """N04: 非負性。"""
    for k, v in values.items():
        if v < 0:
            raise AssertionError(f"N04 violated: {k}={v:.6g}")


def check_no_double_consumption(
    source_initial_mol: float,
    outflows_mol: Sequence[float],
    source_final_mol: float,
    tol_mol: float,
) -> None:
    """N08: 共通原料の二重消費防止。"""
    expect = source_initial_mol - sum(outflows_mol)
    if abs(expect - source_final_mol) > tol_mol:
        raise AssertionError(
            f"N08 violated: expected={expect:.6g} actual={source_final_mol:.6g}"
        )
    if source_final_mol < -tol_mol:
        raise AssertionError("N08 violated: negative source inventory")


def check_dissipation(dissipation_W: float, tol_W: float = 0.0) -> float:
    """N10: 適用条件下での熱力学的散逸 (Σq·Δμ ≥ 0)。"""
    if dissipation_W < -tol_W:
        raise AssertionError(f"N10 violated: dissipation={dissipation_W:.6g} W")
    return dissipation_W


# ---------------------------------------------------------------------------
# §21.3-21.4 識別・不一致の指針
# ---------------------------------------------------------------------------


IDENTIFICATION_GUIDE = {
    "single_profile": "一つの拡散プロファイルから多元拡散行列全体を一意推定しない (§21.3)",
    "combine": ["下位系データ", "異なる勾配方向", "制御配置", "異なる時間条件"],
    "contract_when_unidentifiable": ["有効係数", "組合せパラメータ", "予測可能な観測量"],
}

MISFIT_RESPONSES = [
    "適用範囲縮小",
    "競合機構追加",
    "初期状態の再評価",
    "観測モデルの再評価",
    "識別実験",
]


def forward_validation_protocol(
    calibration_conditions: Sequence[str],
    validation_conditions: Sequence[str],
) -> Dict[str, Any]:
    """前向き実験検証の手順 (§25.5)。検証対象と校正対象を実験前に固定する。"""
    overlap = set(calibration_conditions) & set(validation_conditions)
    if overlap:
        raise ValueError(f"校正と検証の条件が重複: {overlap} (§25.5)")
    return {
        "calibration": list(calibration_conditions),
        "validation": list(validation_conditions),
        "dimensions": ["保持時間", "配置", "供給距離", "配合", "独立作製試料", "別バッチ"],
        "note": "作製前予測と作製後条件付き予測は別々に評価する (§8.5)",
    }
