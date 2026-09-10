"""§10 保存的計算の試験: N01-N04, N07-N09。"""
import pytest

from sdm_react.inventory import (
    CouplingMode,
    FixedStoichiometry,
    InventoryLedger,
    OverdraftPolicy,
    VariableCompositionPhase,
    check_coupling,
)
from sdm_react.validation import (
    check_antisymmetry,
    check_closed_conservation,
    check_no_double_consumption,
    check_nonnegative,
    check_open_balance,
)


def _ledger():
    led = InventoryLedger(["src", "mid", "tgt"], ["Al", "Cu"])
    led.set("src", "Al", 10.0)
    led.set("src", "Cu", 5.0)
    led.set("mid", "Al", 1.0)
    return led


def test_n01_closed_conservation():
    led = _ledger()
    init = {el: led.total(el) for el in led.elements}
    led.transfer("src", "mid", "Al", 3.0)
    led.transfer("mid", "tgt", "Cu", 0.0)
    led.transfer("mid", "tgt", "Al", 1.5)
    res = led.check_conservation(init, tol_mol=1e-12)
    assert all(abs(v) < 1e-12 for v in res.values())
    check_closed_conservation(init, {el: led.total(el) for el in led.elements}, 1e-12)


def test_n02_open_balance():
    led = _ledger()
    init = {el: led.total(el) for el in led.elements}
    led.external_supply("tgt", "Al", 2.0)
    led.transfer("src", "tgt", "Al", 1.0)
    res = led.check_conservation(init, tol_mol=1e-12)
    assert res["Al"] == pytest.approx(0.0)
    check_open_balance(
        init,
        {el: led.total(el) for el in led.elements},
        {"Al": 2.0, "Cu": 0.0},
        1e-12,
    )


def test_n03_antisymmetry_and_n04_nonnegative():
    led = _ledger()
    moved = led.transfer("src", "mid", "Al", 2.0)
    assert moved == pytest.approx(2.0)
    check_antisymmetry(moved, -moved, 1e-12)
    led.check_nonnegative()
    snap = led.snapshot()
    check_nonnegative({f"{r}.{el}": v for r, d in snap.items() for el, v in d.items()})
    with pytest.raises(ValueError):
        led.set("src", "Al", -1.0)


def test_overdraft_raises_by_default_no_clip():
    led = _ledger()
    with pytest.raises(ValueError):
        led.transfer("src", "mid", "Al", 100.0)  # 負在庫→切上げを標準としない (§10.3)


def test_n07_depletion_coupled_outflow():
    # N07: 有限供給源の枯渇 — 要求合計が在庫を超えたら按分し枯渇
    led = _ledger()
    moved = led.coupled_outflow("src", "Al", {"mid": 8.0, "tgt": 8.0})
    assert sum(moved.values()) == pytest.approx(10.0)  # 全在庫のみ流出
    assert led.get("src", "Al") == pytest.approx(0.0)
    assert led.audit  # 縮小率を監査記録


def test_n08_no_double_consumption():
    led = _ledger()
    moved = led.coupled_outflow("src", "Cu", {"mid": 2.0, "tgt": 2.0})
    check_no_double_consumption(5.0, list(moved.values()), led.get("src", "Cu"), 1e-12)


def test_n09_coupling_checklist():
    ok = check_coupling(
        coarse_updated_inside_detail=False,
        boundary_flux_matched=True,
        inventory_transferred=True,
        time_window_matched=True,
        reactions_added_once=True,
    )
    assert ok.all_ok()
    bad = check_coupling(
        coarse_updated_inside_detail=True,  # 置換方式で粗更新が残る = 二重加算
        boundary_flux_matched=True,
        inventory_transferred=True,
        time_window_matched=True,
        reactions_added_once=True,
    )
    assert not bad.all_ok()


def test_fixed_stoichiometry_and_variable_phase():
    sto = FixedStoichiometry({"P": {"Al": 2.0, "Cu": 1.0}})
    assert sto.elements_from_phases({"P": 3.0}) == {"Al": 6.0, "Cu": 3.0}
    assert sto.amount_unit == "mol" and sto.element_unit == "mol"
    vp = VariableCompositionPhase("ss", {"Al": 1.0, "Cu": 4.0})
    assert vp.total_moles() == pytest.approx(5.0)


def test_regions_must_be_unique():
    with pytest.raises(ValueError):
        InventoryLedger(["a", "a"], ["Al"])
