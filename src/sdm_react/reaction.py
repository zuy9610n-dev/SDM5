"""ReactionSolver: 有限供給・反応経路・熱履歴の予測 (設計書 §10-§13)。

水準 M0-M4 (§11)、供給経路・元素参加時刻 (§12)、情報取得時間窓 (§13)。
保存則・適用範囲・出力能力を明確にした複数モデルを目的に応じて接続する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sdm_react.core import (
    BaseModel,
    Capability,
    DomainStatus,
    ExecStatus,
    IdentStatus,
    ModelOutput,
    OutputNature,
    ResultStatus,
)
from sdm_react.inventory import InventoryLedger


def _trapz(y: Sequence[float], x: Sequence[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return area


# ---------------------------------------------------------------------------
# §11.2 M0: 供給スクリーニング
# ---------------------------------------------------------------------------


class CoefficientKind(str, Enum):
    SELF = "self"  # 原子の自己拡散
    IN_PHASE = "in_phase"  # 指定相内の輸送
    EFFECTIVE = "effective"  # 複数界面を含む有効輸送


class M0Screening(BaseModel):
    """輸送尺度 ell ~ sqrt(∫D_eff dt) による幾何学的・供給的スクリーニング。

    出力を定量組成場や相生成予測へ昇格させない (§11.2)。
    """

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        missing = []
        for k in ("time_s", "D_eff_m2_s", "coefficient_kind"):
            if k not in inputs:
                missing.append(k)
        return missing

    def describe_capabilities(self) -> Capability:
        return Capability(
            name="M0:supply_screening",
            required_inputs=["time_s", "D_eff_m2_s", "coefficient_kind"],
            state_variables=["ell_m"],
            conserved_quantities=[],
            units={"ell_m": "m", "D_eff": "m^2/s", "time_s": "s"},
            predictable=["輸送尺度", "供給困難度", "ボトルネック候補"],
            unpredictable=["定量組成場", "相生成", "層厚", "組成プロファイル"],
            domain=["スクリーニング専用。係数の意味の混同禁止"],
            uncertainty_form=["scenario"],
            diagnostics=["integral_Ddt"],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return {"use": "screening_only", "forbids": ["composition_field", "phase_prediction"]}

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        missing = self.check_requirements(inputs)
        if missing:
            return ModelOutput(
                None,
                ResultStatus(execution=ExecStatus.INSUFFICIENT_INPUT,
                             notes=[f"missing: {missing}"]),
            )
        t = list(inputs["time_s"])
        kind = CoefficientKind(inputs["coefficient_kind"])
        ell: Dict[str, float] = {}
        integrals: Dict[str, float] = {}
        for el, dser in inputs["D_eff_m2_s"].items():
            d = list(dser)
            if len(d) != len(t):
                return ModelOutput(
                    None,
                    ResultStatus(execution=ExecStatus.NUMERICAL_FAILURE,
                                 notes=[f"length mismatch for {el}"]),
                )
            integ = _trapz(d, t)
            integrals[el] = integ
            ell[el] = math.sqrt(max(integ, 0.0))
        return ModelOutput(
            {"ell_m": ell, "coefficient_kind": kind.value},
            ResultStatus(
                domain=DomainStatus.CALIBRATED_ONLY,
                nature=OutputNature.OUTER_BOUND,
                notes=[
                    f"係数種別={kind.value}。自己拡散/相内輸送/有効輸送を混同しない (§11.2)",
                    "定量組成場・相生成予測への昇格禁止",
                ],
            ),
            {"integral_Ddt_m2": integrals},
        )

    def compose_field(self, *a: Any, **k: Any) -> None:
        """独立到達指標の規格化による組成場出力を拒否 (§2.2 禁止事項)。"""
        raise NotImplementedError(
            "M0 から組成場を生成してはならない (§2.2, §11.2)。"
        )


# ---------------------------------------------------------------------------
# §11.3 M1: 有効反応モデル
# ---------------------------------------------------------------------------


class M1Law(str, Enum):
    PARABOLIC = "parabolic"  # xi^2 - xi0^2 = K(t - t0)
    LINEAR_PARABOLIC = "linear_parabolic"  # t - t0 = A(xi-xi0) + B(xi^2-xi0^2)


@dataclass
class M1Params:
    law: M1Law
    K_eff_m2_s: float = 0.0  # m^2/s (相互拡散係数と呼ばない)
    A_s_m: float = 0.0  # s/m
    B_s_m2: float = 0.0  # s/m^2
    xi0_m: float = 0.0
    t0_s: float = 0.0
    t0_definition: str = ""
    product_composition: Dict[str, float] = field(default_factory=dict)  # 生成物組成 (原子分率)
    product_molar_volume_m3_mol: float = 0.0
    contact_area_m2: float = 0.0
    domain: Dict[str, Any] = field(default_factory=dict)


class M1EffectiveReaction(BaseModel):
    """層成長の有効反応モデル (§11.3)。

    供給枯渇後は成長式を延長しない。層厚の上限切りと元素消費を必ず接続する。
    """

    def __init__(self, params: M1Params) -> None:
        self.p = params

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        missing = []
        if "time_s" not in inputs:
            missing.append("time_s")
        if "available_mol" not in inputs:
            missing.append("available_mol")
        if not self.p.t0_definition:
            missing.append("t0_definition(反応開始時点の定義, §11.3)")
        if not self.p.product_composition:
            missing.append("product_composition(生成物組成, §11.3)")
        if self.p.product_molar_volume_m3_mol <= 0:
            missing.append("product_molar_volume(層体積と元素消費の接続, §11.3)")
        if self.p.contact_area_m2 <= 0:
            missing.append("contact_area_m2")
        return missing

    def describe_capabilities(self) -> Capability:
        return Capability(
            name="M1:effective_reaction",
            required_inputs=["time_s", "available_mol"],
            state_variables=["xi_m", "consumed_mol"],
            conserved_quantities=["element_inventory"],
            units={"xi_m": "m", "K_eff": "m^2/s", "A": "s/m", "B": "s/m^2"},
            predictable=["層厚", "消費量", "成長停止", "遅延"],
            unpredictable=["層内濃度分布", "多元拡散行列", "核生成"],
            domain=["適用温度・時間・形状内のみ", "K_eff は有効量"],
            uncertainty_form=["scenario", "interval"],
            diagnostics=["xi_unlimited_m", "xi_max_depletion_m", "limiting_element"],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return dict(self.p.domain) or {"note": "適用温度・時間・形状を呼び出し側で指定"}

    def _growth_law(self, t: float) -> float:
        p = self.p
        if t <= p.t0_s:
            return p.xi0_m
        if p.law == M1Law.PARABOLIC:
            return math.sqrt(max(p.xi0_m**2 + p.K_eff_m2_s * (t - p.t0_s), 0.0))
        # B xi^2 + A xi - (A xi0 + B xi0^2 + dt) = 0 の正根
        dt = t - p.t0_s
        c = -(p.A_s_m * p.xi0_m + p.B_s_m2 * p.xi0_m**2 + dt)
        if abs(p.B_s_m2) < 1e-30:
            if abs(p.A_s_m) < 1e-30:
                return p.xi0_m
            return max(p.xi0_m + dt / p.A_s_m, p.xi0_m)
        disc = p.A_s_m**2 - 4.0 * p.B_s_m2 * c
        if disc < 0:
            return p.xi0_m
        root = (-p.A_s_m + math.sqrt(disc)) / (2.0 * p.B_s_m2)
        return max(root, p.xi0_m)

    def _depletion_limit(self, available_mol: Mapping[str, float]) -> Tuple[float, str]:
        """供給枯渇による層厚上限と律速元素 (§11.3)。"""
        p = self.p
        Vm = p.product_molar_volume_m3_mol
        A = p.contact_area_m2
        xi_max = float("inf")
        limiting = ""
        for el, x in p.product_composition.items():
            if x <= 0:
                continue
            c_el = x / Vm  # mol/m^3
            xi_el = float(available_mol.get(el, 0.0)) / (A * c_el)
            if xi_el < xi_max:
                xi_max = xi_el
                limiting = el
        return xi_max, limiting

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        missing = self.check_requirements(inputs)
        if missing:
            return ModelOutput(
                None,
                ResultStatus(execution=ExecStatus.INSUFFICIENT_INPUT,
                             notes=[f"missing: {missing}"]),
            )
        times = list(inputs["time_s"])
        avail = dict(inputs["available_mol"])
        Vm = self.p.product_molar_volume_m3_mol
        A = self.p.contact_area_m2
        xi_max, limiting = self._depletion_limit(avail)
        xi_list, cons_list, capped = [], [], []
        for t in times:
            xi_free = self._growth_law(float(t))
            xi = min(xi_free, xi_max)
            capped.append(xi_free > xi_max)
            vol = A * xi
            n_prod = vol / Vm
            cons = {el: n_prod * x for el, x in self.p.product_composition.items()}
            xi_list.append(xi)
            cons_list.append(cons)
        notes = [
            "K_eff は有効成長係数であり相互拡散係数と呼ばない (§11.3)",
            f"枯渇律速元素={limiting}, xi_max={xi_max:.6g} m。枯渇後の延長なし",
        ]
        if any(capped):
            notes.append("一部の時刻で枯渇上限が作用。層厚と消費量は整合的に停止")
        return ModelOutput(
            {"time_s": times, "xi_m": xi_list, "consumed_mol": cons_list},
            ResultStatus(
                domain=DomainStatus.CALIBRATED_ONLY,
                nature=OutputNature.PHYSICAL_PREDICTION,
                notes=notes,
            ),
            {"xi_unlimited_m": [self._growth_law(float(t)) for t in times],
             "xi_max_depletion_m": xi_max, "limiting_element": limiting,
             "capped": capped},
        )


# ---------------------------------------------------------------------------
# §11.4-11.5 M2: 多元輸送 (簡略ネットワーク輸送)
# ---------------------------------------------------------------------------


@dataclass
class TransportEdge:
    a: str
    b: str
    K: List[List[float]]  # 対称・半正定値の輸送行列 (§11.5)
    independent_elements: List[str]


def _is_symmetric_psd(K: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    n = len(K)
    for i in range(n):
        for j in range(n):
            if abs(K[i][j] - K[j][i]) > tol:
                return False
    # 2x2 までの解析判定 + 一般は対角優位の簡易確認ではなく固有値を numpy で確認
    try:
        import numpy as np  # type: ignore

        w = np.linalg.eigvalsh(np.array(K, dtype=float))
        return bool((w > -tol).all())
    except ImportError:
        if n == 1:
            return K[0][0] >= -tol
        if n == 2:
            a, b, c = K[0][0], K[0][1], K[1][1]
            return a >= -tol and c >= -tol and a * c - b * b >= -tol
        raise ImportError("3元以上の PSD 判定には numpy が必要")


def ideal_mu_J_mol(
    x: Mapping[str, float],
    mu0_J_mol: Mapping[str, float],
    T_K: float,
    R: float = 8.314462618,
) -> Dict[str, float]:
    """理想混合の例示的化学ポテンシャル。熱力学モデル使用時は由来を記録すること。"""
    out = {}
    for el, xi in x.items():
        xi_c = max(float(xi), 1e-300)
        out[el] = float(mu0_J_mol.get(el, 0.0)) + R * T_K * math.log(xi_c)
    return out


class M2NetworkTransport(BaseModel):
    """簡略ネットワーク輸送 q_ab = K_ab (μ_a - μ_b) (§11.5)。

    有限体積/保存的領域ネットワークによる保存的離散化 (§24.1)。
    参照系・独立成分・モル体積・境界条件を明示する (§11.4)。
    """

    def __init__(
        self,
        edges: List[TransportEdge],
        reference_frame: str,
        molar_volume_m3_mol: float,
        total_molar_flow_zero: bool = True,
    ) -> None:
        self.edges = edges
        self.reference_frame = reference_frame
        self.molar_volume = molar_volume_m3_mol
        self.total_molar_flow_zero = total_molar_flow_zero

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        missing = []
        for k in ("nodes_mol", "mu_J_mol", "dt_s", "elements"):
            if k not in inputs:
                missing.append(k)
        # 追加成分追跡時に主五元素だけの四自由度表示を使わない (§11.4)
        els = list(inputs.get("elements", []))
        from sdm_react.core import ELEMENTS as MAIN5

        extra = [e for e in els if e not in MAIN5]
        for e in self.edges:
            if extra and set(e.independent_elements) <= set(MAIN5) and len(e.independent_elements) == 4:
                missing.append(
                    f"4-dof main-only display forbidden with extra {extra} (§11.4)"
                )
        for e in self.edges:
            if not _is_symmetric_psd(e.K):
                missing.append(f"K must be symmetric PSD (§11.5): {e.a}-{e.b}")
        return missing

    def describe_capabilities(self) -> Capability:
        return Capability(
            name="M2:network_transport",
            required_inputs=["nodes_mol", "mu_J_mol", "dt_s", "elements"],
            state_variables=["n_mol"],
            conserved_quantities=["element_inventory"],
            units={"n": "mol", "mu": "J/mol", "q": "mol/s", "K": "mol^2/(J s)"},
            predictable=["濃度場 (指定相内)", "相内流束"],
            unpredictable=["相生成", "界面移動", "核生成"],
            domain=["等温・固定モル量ノード等の仮定が妥当な範囲 (§11.5)"],
            uncertainty_form=["scenario"],
            diagnostics=["dissipation_W", "dt_limited", "antisymmetry_err"],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return {
            "reference_frame": self.reference_frame,
            "molar_volume_m3_mol": self.molar_volume,
            "total_molar_flow_zero": self.total_molar_flow_zero,
            "note": "体積変化・格子移動・空孔流束を扱ったことにはならない (§11.5)",
        }

    def _fluxes(
        self, mu: Mapping[str, Mapping[str, float]]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """各辺の元素流量 q_ab [mol/s]。"""
        out: Dict[Tuple[str, str], Dict[str, float]] = {}
        for e in self.edges:
            ind = e.independent_elements
            dmu = [mu[e.a][el] - mu[e.b][el] for el in ind]
            n = len(ind)
            q = [sum(e.K[i][j] * dmu[j] for j in range(n)) for i in range(n)]
            qd = {el: q[i] for i, el in enumerate(ind)}
            if self.total_molar_flow_zero:
                # 従属成分は総モル流量ゼロから定める (独立成分以外の残差)
                qd["__dependent__"] = -sum(q)
            out[(e.a, e.b)] = qd
        return out

    def dissipation_W(self, mu: Mapping[str, Mapping[str, float]]) -> float:
        """閉鎖等温系の輸送散逸 Σ q·Δμ ≥ 0 の確認 (§11.5, N10)。"""
        dis = 0.0
        for (a, b), qd in self._fluxes(mu).items():
            edge = next(e for e in self.edges if (e.a, e.b) == (a, b))
            for el in edge.independent_elements:
                dis += qd[el] * (mu[a][el] - mu[b][el])
        return dis

    def step(
        self,
        ledger: InventoryLedger,
        mu_J_mol: Mapping[str, Mapping[str, float]],
        dt_s: float,
        dependent_element: Optional[str] = None,
        interval: int = 0,
    ) -> Dict[str, Any]:
        """1 時間刻みの保存的更新。非負性を保つため必要なら dt を縮小する (§24.2)。"""
        fluxes = self._fluxes(mu_J_mol)
        # 非負性を満たす dt 上限を見積もる
        dt = dt_s
        for (a, b), qd in fluxes.items():
            for el, q in qd.items():
                if el == "__dependent__":
                    continue
                if q > 0:
                    avail = ledger.get(a, el)
                    if q * dt > avail and q > 0:
                        dt = avail / q if avail > 0 else 0.0
                elif q < 0:
                    avail = ledger.get(b, el)
                    if -q * dt > avail and -q > 0:
                        dt = avail / (-q) if avail > 0 else 0.0
        limited = dt < dt_s
        for (a, b), qd in fluxes.items():
            for el, q in qd.items():
                if el == "__dependent__":
                    if dependent_element is None:
                        continue
                    el_use = dependent_element
                else:
                    el_use = el
                amt = abs(q) * dt
                if amt == 0:
                    continue
                if q > 0:
                    ledger.transfer(a, b, el_use, amt, interval)
                else:
                    ledger.transfer(b, a, el_use, amt, interval)
        return {"dt_used_s": dt, "dt_limited": limited,
                "dissipation_W": self.dissipation_W(mu_J_mol)}

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        return ModelOutput(
            None,
            ResultStatus(
                execution=ExecStatus.INSUFFICIENT_INPUT,
                notes=["M2 は step() による刻み更新で使用する (台帳結合のため)"],
            ),
        )


# ---------------------------------------------------------------------------
# §11.6 M3: 移動界面・多相反応
# ---------------------------------------------------------------------------


class InterfaceMode(str, Enum):
    LOCAL_EQUILIBRIUM = "local_equilibrium"
    FINITE_RATE = "finite_rate"


@dataclass
class PhaseCandidateLog:
    candidates: List[str] = field(default_factory=list)
    nucleation_conditions: Dict[str, str] = field(default_factory=dict)
    suppressed: Dict[str, str] = field(default_factory=dict)  # 相 -> 抑制理由


class M3MovingInterface(BaseModel):
    """界面跳躍条件の検査器 (§11.6)。符号規約を固定する。

    n: a→b 方向の界面法線。[Q] = Q_b - Q_a。
    [J_i·n] = v_n [c_i]。
    """

    def __init__(self, mode: InterfaceMode, phase_log: Optional[PhaseCandidateLog] = None) -> None:
        self.mode = mode
        self.phase_log = phase_log or PhaseCandidateLog()

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        missing = []
        for k in ("J_a", "J_b", "c_a", "c_b", "v_n"):
            if k not in inputs:
                missing.append(k)
        return missing

    def describe_capabilities(self) -> Capability:
        return Capability(
            name=f"M3:moving_interface:{self.mode.value}",
            required_inputs=["J_a", "J_b", "c_a", "c_b", "v_n"],
            state_variables=["jump_residual"],
            conserved_quantities=["element_inventory"],
            units={"J": "mol/(m^2 s)", "c": "mol/m^3", "v_n": "m/s"},
            predictable=["界面跳躍の整合性", "界面移動速度の整合性"],
            unpredictable=["核生成時刻", "安定相の時間内生成の保証"],
            domain=["指定相集合の多相反応"],
            uncertainty_form=["scenario"],
            diagnostics=["jump_residual"],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "sign_convention": "[Q]=Q_b-Q_a, n:a->b",
            "note": "熱力学的安定相の存在は時間内生成の保証ではない (§11.6)",
        }

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        missing = self.check_requirements(inputs)
        if missing:
            return ModelOutput(
                None,
                ResultStatus(execution=ExecStatus.INSUFFICIENT_INPUT,
                             notes=[f"missing: {missing}"]),
            )
        Ja = dict(inputs["J_a"])
        Jb = dict(inputs["J_b"])
        ca = dict(inputs["c_a"])
        cb = dict(inputs["c_b"])
        vn = float(inputs["v_n"])
        els = set(Ja) | set(Jb) | set(ca) | set(cb)
        resid = {el: (Jb.get(el, 0.0) - Ja.get(el, 0.0)) - vn * (cb.get(el, 0.0) - ca.get(el, 0.0))
                 for el in els}
        return ModelOutput(
            {"jump_residual": resid},
            ResultStatus(nature=OutputNature.PHYSICAL_PREDICTION,
                         notes=[f"mode={self.mode.value}。局所平衡と有限反応速度は別モデル (§11.6)"]),
            {"jump_residual": resid},
        )


# ---------------------------------------------------------------------------
# §11.7 M4: 特殊機構 (目的別の小モデルから開始)
# ---------------------------------------------------------------------------


class M4Infiltration(BaseModel):
    """経験的浸透距離モデル (§11.7)。全面 3D 連成の代替ではない。"""

    def __init__(self, k_m_s05: float, t_ref_s: float = 0.0) -> None:
        self.k = k_m_s05
        self.t_ref = t_ref_s

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        return [] if "time_s" in inputs else ["time_s"]

    def describe_capabilities(self) -> Capability:
        return Capability(
            name="M4:infiltration",
            required_inputs=["time_s"],
            state_variables=["infiltration_m"],
            conserved_quantities=[],
            units={"infiltration_m": "m"},
            predictable=["経験的浸透距離"],
            unpredictable=["組成場", "相", "濡れ角"],
            domain=["校正範囲内のみ"],
            uncertainty_form=["interval"],
            diagnostics=[],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return {"empirical": True, "form": "d = k*sqrt(t - t_ref)"}

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        return ModelOutput(
            {"infiltration_m": [self.k * math.sqrt(max(float(t) - self.t_ref, 0.0))
                                for t in inputs["time_s"]]},
            ResultStatus(nature=OutputNature.EMPIRICAL_PREDICTION,
                         notes=["経験的浸透距離。対象機構外条件の推奨に使わない (§11.7)"]),
        )


class M4InterfaceResistance(BaseModel):
    """界面抵抗モデル: q = Δμ / R (§11.7)。"""

    def __init__(self, R_J_s_mol2: float) -> None:
        self.R = R_J_s_mol2

    def check_requirements(self, inputs: Mapping[str, Any]) -> List[str]:
        return [] if "dmu_J_mol" in inputs else ["dmu_J_mol"]

    def describe_capabilities(self) -> Capability:
        return Capability(
            name="M4:interface_resistance",
            required_inputs=["dmu_J_mol"],
            state_variables=["q_mol_s"],
            conserved_quantities=["element_inventory"],
            units={"q": "mol/s", "R": "J s/mol^2"},
            predictable=["界面通過流量"],
            unpredictable=["界面構造", "組成分布"],
            domain=["校正範囲内のみ"],
            uncertainty_form=["interval"],
            diagnostics=[],
        )

    def describe_domain(self) -> Dict[str, Any]:
        return {"empirical": True, "form": "q = dmu / R"}

    def predict(self, inputs: Mapping[str, Any]) -> ModelOutput:
        dmu = dict(inputs["dmu_J_mol"])
        return ModelOutput(
            {"q_mol_s": {el: v / self.R for el, v in dmu.items()}},
            ResultStatus(nature=OutputNature.EMPIRICAL_PREDICTION),
        )


# ---------------------------------------------------------------------------
# §12 供給経路と元素参加時刻
# ---------------------------------------------------------------------------


def net_flow_paths(ledger: InventoryLedger) -> List[Dict[str, Any]]:
    """P1: 正味流量経路。保存モデルから得られる領域間の正味元素移送 (§12.2)。"""
    net = ledger.net_flows()
    paths = [
        {"from": a, "to": b, "element": el, "amount_mol": amt}
        for (a, b, el), amt in sorted(net.items(), key=lambda kv: -abs(kv[1]))
    ]
    return paths


def virtual_intervention_sensitivity(
    run_base: Callable[[], Dict[str, float]],
    run_modified: Callable[[], Dict[str, float]],
) -> Dict[str, float]:
    """P2: 仮想介入感度。供給源の位置・量・接触変更時の予測差 (§12.2)。"""
    base = run_base()
    mod = run_modified()
    keys = set(base) | set(mod)
    return {k: mod.get(k, 0.0) - base.get(k, 0.0) for k in keys}


def source_provenance_without_tracer(*a: Any, **k: Any) -> None:
    """P3: トレーサーなしの供給源由来追跡を拒否 (§12.2)。

    P3 を総流束の単純按分で生成しない。初期実装は P1・P2 を優先する。
    """
    raise NotImplementedError(
        "供給源由来 (P3) にはトレーサー輸送等の対応モデルが必要。"
        "総流束の単純按分で生成してはならない (§12.2)。"
    )


@dataclass
class EntryTime:
    element: str
    t_entry_s: Optional[float]
    threshold_mol: float
    threshold_basis: str  # 物理的/観測的根拠の記述 (必須)
    target_kind: str  # 何への参加か (領域量/相取込/定量可能性...)
    preexisting: bool = False
    notes: List[str] = field(default_factory=list)


def entry_time(
    element: str,
    time_s: Sequence[float],
    amount_mol: Sequence[float],
    threshold_mol: float,
    threshold_basis: str,
    target_kind: str,
) -> EntryTime:
    """元素参加時刻 t_entry = inf{t: n_R(t) ≥ n*} (§12.3)。

    何への参加かを明示する。初期在存の場合は到着時刻を表さない。
    """
    if not threshold_basis or len(threshold_basis.strip()) < 5:
        raise ValueError("閾値 n* には物理的または観測的根拠が必要 (§12.3)")
    t_list = list(time_s)
    n_list = list(amount_mol)
    preexisting = bool(n_list and n_list[0] >= threshold_mol)
    t_hit = None
    for t, n in zip(t_list, n_list):
        if n >= threshold_mol:
            t_hit = float(t)
            break
    notes = []
    if preexisting:
        notes.append("初期から領域内に存在するため新たな供給の到着時刻を表さない (§12.3)")
    return EntryTime(element, t_hit, threshold_mol, threshold_basis, target_kind,
                     preexisting, notes)


# ---------------------------------------------------------------------------
# §13 情報取得時間窓
# ---------------------------------------------------------------------------


def passing_intervals(
    time_s: Sequence[float], verdicts: Sequence[str]
) -> List[Tuple[float, float]]:
    """合格時刻集合 T_q(d,A) を区間列として抽出する (非連続可, §13.1)。"""
    intervals: List[Tuple[float, float]] = []
    start: Optional[float] = None
    prev = 0.0
    for t, v in zip(time_s, verdicts):
        if v == "pass" and start is None:
            start = float(t)
        if v != "pass" and start is not None:
            intervals.append((start, prev))
            start = None
        prev = float(t)
    if start is not None:
        intervals.append((start, prev))
    return intervals


def robust_intersection(
    scenarios: Mapping[str, List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """頑健な時間窓: 全採用シナリオの共通合格時刻 (∩, §13.2)。

    空集合は頑健な合格条件が見つからないことを意味し、
    目標そのものの不可能性とは限らない。
    """
    lists = list(scenarios.values())
    if not lists:
        return []
    result = list(lists[0])
    for other in lists[1:]:
        new: List[Tuple[float, float]] = []
        for a0, a1 in result:
            for b0, b1 in other:
                lo, hi = max(a0, b0), min(a1, b1)
                if hi >= lo:
                    new.append((lo, hi))
        result = new
    return result
