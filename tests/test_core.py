"""§4 数量・状態・三値判定、N05 換算の試験。"""
import math

import pytest

from sdm_react.core import (
    CompositionRecord,
    MaterialState,
    combine_verdicts,
    mass_fractions_to_mole_fractions,
    mean_composition,
    mixture_volume_additive_m3,
    mol_to_mass_kg,
    mass_kg_to_mol,
    mole_fractions,
    mole_fractions_to_mass_fractions,
    pure_molar_volume_m3_mol,
)


def test_combine_verdicts():
    assert combine_verdicts(["pass", "pass"]) == "pass"
    assert combine_verdicts(["pass", "fail"]) == "fail"
    assert combine_verdicts(["pass", "unknown"]) == "unknown"
    assert combine_verdicts(["unknown"]) == "unknown"


def test_mole_fractions_sum_to_one():
    x = mole_fractions({"Al": 2.0, "Cu": 3.0})
    assert x == pytest.approx({"Al": 0.4, "Cu": 0.6})
    with pytest.raises(ValueError):
        mole_fractions({"Al": 0.0})


def test_mean_composition_is_inventory_based():
    # 元素量基準の平均 (§4.2)
    m = mean_composition([{"Al": 1.0, "Cu": 1.0}, {"Al": 3.0, "Cu": 1.0}])
    assert m["Al"] == pytest.approx(4.0 / 6.0)
    assert m["Cu"] == pytest.approx(2.0 / 6.0)


def test_n05_roundtrip_mol_mass_volume():
    # N05: 原子・質量・体積換算の往復
    for el in ("Al", "Si", "Ti", "Fe", "Cu"):
        m = mol_to_mass_kg(2.5, el)
        assert mass_kg_to_mol(m, el) == pytest.approx(2.5)
        assert pure_molar_volume_m3_mol(el) > 0
    x = {"Al": 0.2, "Si": 0.2, "Ti": 0.2, "Fe": 0.2, "Cu": 0.2}
    w = mole_fractions_to_mass_fractions(x)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert mass_fractions_to_mole_fractions(w) == pytest.approx(x)
    vol = mixture_volume_additive_m3({"Al": 1.0, "Cu": 1.0})
    assert vol == pytest.approx(
        pure_molar_volume_m3_mol("Al") + pure_molar_volume_m3_mol("Cu")
    )


def test_material_state_missing_is_explicit():
    s = MaterialState(temperature_K=1000.0)
    missing = s.missing()
    assert "temperature_K" not in missing
    assert "phases" in missing  # 未予測は欠測として保持 (§4.1)


def test_composition_record_keeps_raw():
    rec = CompositionRecord.from_raw({"Al": 0.5, "Cu": 0.4, "O": 0.1})
    assert rec.raw_total == pytest.approx(1.0)
    assert set(rec.raw_values) == {"Al", "Cu", "O"}
    assert rec.renormalized_5el is not None
    assert rec.extra_measured["O"] is True
    # 再規格化値だけでなく元の分析値・合計を保持 (§4.3)
    assert rec.raw_values["O"] == pytest.approx(0.1)
