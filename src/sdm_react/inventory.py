"""有限供給・反応の保存的計算: 元素在庫の一元管理 (設計書 §10, §24)。

- 計算領域は重複しない所有領域に分ける
- 内部輸送は反対称 Φ_ab = -Φ_ba
- 閉鎖系では Σ_a n の時間微分 = 0
- 負在庫→ゼロ切上げを標準の保存法としない (§10.3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OverdraftPolicy(str, Enum):
    RAISE = "raise"  # 既定: 超過を例外にする
    CLIP_WITH_AUDIT = "clip_with_audit"  # 明示的監査付きでのみ許可


@dataclass
class FlowRecord:
    frm: str
    to: str
    element: str
    amount_mol: float
    interval: int = 0


class InventoryLedger:
    """元素在庫の唯一の台帳 (§10.1)。"""

    def __init__(
        self,
        regions: List[str],
        elements: List[str],
        overdraft: OverdraftPolicy = OverdraftPolicy.RAISE,
    ) -> None:
        if len(set(regions)) != len(regions):
            raise ValueError("regions must be non-overlapping (unique) (§10.1)")
        self.regions = list(regions)
        self.elements = list(elements)
        self.overdraft = overdraft
        self._n: Dict[Tuple[str, str], float] = {
            (el, r): 0.0 for el in elements for r in regions
        }
        self.flows: List[FlowRecord] = []
        self.external_inflow_mol: Dict[Tuple[str, str], float] = {}
        self.audit: List[str] = []

    # -- 設定・参照 ------------------------------------------------------
    def set(self, region: str, element: str, moles: float) -> None:
        self._check(region, element)
        if moles < 0:
            raise ValueError("inventory must be non-negative (§10.1)")
        self._n[(element, region)] = float(moles)

    def get(self, region: str, element: str) -> float:
        self._check(region, element)
        return self._n[(element, region)]

    def total(self, element: str) -> float:
        if element not in self.elements:
            raise KeyError(element)
        return sum(self._n[(element, r)] for r in self.regions)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            r: {el: self._n[(el, r)] for el in self.elements} for r in self.regions
        }

    # -- 更新 ------------------------------------------------------------
    def transfer(
        self,
        frm: str,
        to: str,
        element: str,
        amount_mol: float,
        interval: int = 0,
    ) -> float:
        """内部輸送。反対称に在庫を更新する (§10.1)。

        戻り値は実際に移送した量。超過時は方針に従う。
        """
        self._check(frm, element)
        self._check(to, element)
        if amount_mol < 0:
            raise ValueError("transfer amount must be non-negative")
        avail = self._n[(element, frm)]
        moved = amount_mol
        if moved > avail:
            if self.overdraft == OverdraftPolicy.RAISE:
                raise ValueError(
                    f"overdraft: {element} in {frm}: "
                    f"request={moved:.6g} > avail={avail:.6g} (§10.3)"
                )
            moved = avail
            self.audit.append(
                f"CLIP interval={interval} {frm}->{to} {element}: "
                f"request={amount_mol:.6g} avail={avail:.6g}"
            )
        self._n[(element, frm)] = avail - moved
        self._n[(element, to)] = self._n[(element, to)] + moved
        self.flows.append(FlowRecord(frm, to, element, moved, interval))
        return moved

    def external_supply(
        self, region: str, element: str, moles: float, interval: int = 0
    ) -> None:
        """外部からの正味供給 S^{external} (§10.1)。"""
        self._check(region, element)
        self._n[(element, region)] += float(moles)
        key = (element, region)
        self.external_inflow_mol[key] = (
            self.external_inflow_mol.get(key, 0.0) + float(moles)
        )
        if self._n[(element, region)] < 0:
            raise ValueError("external supply drove inventory negative")

    def coupled_outflow(
        self,
        source: str,
        element: str,
        requests: Dict[str, float],
        interval: int = 0,
    ) -> Dict[str, float]:
        """共通供給源からの連成流出 (§10.3)。

        複数反応領域が同じ原料を消費する場合、各局所モデルへ原料全量を
        独立に与えない。時間刻み内の合計流出を在庫と整合させる。
        不足時は按分 (縮小率を監査記録)。
        """
        self._check(source, element)
        total_req = sum(requests.values())
        if total_req < 0:
            raise ValueError("requests must be non-negative")
        avail = self._n[(element, source)]
        scale = 1.0 if total_req <= avail else (avail / total_req if total_req > 0 else 1.0)
        if scale < 1.0:
            self.audit.append(
                f"COUPLED interval={interval} source={source} {element}: "
                f"total_request={total_req:.6g} avail={avail:.6g} scale={scale:.6g}"
            )
        moved: Dict[str, float] = {}
        for dest, req in requests.items():
            moved[dest] = self.transfer(source, dest, element, req * scale, interval)
        return moved

    # -- 検証 ------------------------------------------------------------
    def check_conservation(
        self, initial_totals: Dict[str, float], tol_mol: float = 0.0
    ) -> Dict[str, float]:
        """閉鎖系の保存確認。外部流入がある場合はそれを加味する (N01, N02)。

        戻り値: 元素別の残差 (現在総量 - 初期総量 - 外部正味流入)。
        """
        residuals = {}
        for el in self.elements:
            ext = sum(
                v for (e, _r), v in self.external_inflow_mol.items() if e == el
            )
            residuals[el] = self.total(el) - initial_totals.get(el, 0.0) - ext
            if abs(residuals[el]) > tol_mol:
                raise AssertionError(
                    f"conservation violated for {el}: residual={residuals[el]:.6g} "
                    f"(tol={tol_mol:.6g})"
                )
        return residuals

    def check_nonnegative(self) -> None:
        """非負性 (N04)。"""
        for key, v in self._n.items():
            if v < 0:
                raise AssertionError(f"negative inventory {key}: {v}")

    def net_flows(self) -> Dict[Tuple[str, str, str], float]:
        """領域間・元素別の正味移送量 (P1 経路解析用, §12.2)。"""
        net: Dict[Tuple[str, str, str], float] = {}
        for f in self.flows:
            key = (f.frm, f.to, f.element)
            net[key] = net.get(key, 0.0) + f.amount_mol
        return net

    def _check(self, region: str, element: str) -> None:
        if region not in self.regions:
            raise KeyError(f"unknown region: {region}")
        if element not in self.elements:
            raise KeyError(f"unknown element: {element}")


# ---------------------------------------------------------------------------
# §10.2 相反応の扱い
# ---------------------------------------------------------------------------


@dataclass
class FixedStoichiometry:
    """固定化学量論の相量-元素量関係 n_i = Σ_p A_ip m_p (§10.2)。

    A_ip の定義と単位を固定する。固溶体には使えない。
    """

    matrix: Dict[str, Dict[str, float]]  # phase -> {element: A_ip}
    amount_unit: str = "mol"  # m_p の単位
    element_unit: str = "mol"  # n_i の単位

    def elements_from_phases(self, phase_amounts: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p, m in phase_amounts.items():
            for el, a in self.matrix[p].items():
                out[el] = out.get(el, 0.0) + a * float(m)
        return out


@dataclass
class VariableCompositionPhase:
    """固溶体など組成可変相: 相ごとの元素在庫を持つ (§10.2)。

    固定化学量論行列だけでは扱わない。
    """

    name: str
    inventory_mol: Dict[str, float] = field(default_factory=dict)

    def total_moles(self) -> float:
        return float(sum(self.inventory_mol.values()))


# ---------------------------------------------------------------------------
# §10.4 全体モデルと局所詳細モデル
# ---------------------------------------------------------------------------


class CouplingMode(str, Enum):
    SUBSTITUTION = "substitution"  # 置換方式
    BOUNDARY_FLUX = "boundary_flux"  # 境界結合方式


@dataclass
class CouplingChecklist:
    """粗密結合の整合項目。二重加算の防止 (§10.4)。"""

    ownership_ok: bool = False
    boundary_flux_ok: bool = False
    inventory_ok: bool = False
    time_window_ok: bool = False
    no_double_counting_ok: bool = False

    def all_ok(self) -> bool:
        return (
            self.ownership_ok
            and self.boundary_flux_ok
            and self.inventory_ok
            and self.time_window_ok
            and self.no_double_counting_ok
        )


def check_coupling(
    coarse_updated_inside_detail: bool,
    boundary_flux_matched: bool,
    inventory_transferred: bool,
    time_window_matched: bool,
    reactions_added_once: bool,
) -> CouplingChecklist:
    """結合方式の検証。置換方式では詳細領域内の粗い更新を停止すること。"""
    return CouplingChecklist(
        ownership_ok=True,  # 所有権は台帳の一意性で保証
        boundary_flux_ok=bool(boundary_flux_matched),
        inventory_ok=bool(inventory_transferred),
        time_window_ok=bool(time_window_matched),
        no_double_counting_ok=bool(
            (not coarse_updated_inside_detail) and reactions_added_once
        ),
    )
