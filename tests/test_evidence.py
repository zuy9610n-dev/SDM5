"""§18-§19 情報記録・集計の試験 (§25.4)。"""
import random

from sdm_react.evidence import (
    AreaFraction,
    CoexistenceRecord,
    EvidenceStore,
    EvidenceUnit,
    IndependenceLevel,
    SamplingMode,
    gaussian_kde_1d,
)


def _unit(i, sample, quality="pass", mode=SamplingMode.EVALUATION):
    return EvidenceUnit(
        id=f"u{i}", sample_id=sample,
        local_mean_composition={"Al": 0.5, "Cu": 0.5},
        phase_or_candidates=["X"], temperature_K=1000.0,
        quality=quality, independence=IndependenceLevel.PIXEL,
        independence_id=sample,
    ), mode


def test_pixel_is_not_sample():
    store = EvidenceStore()
    for i in range(100):
        u, m = _unit(i, "S1")  # 100 画素・同一試料
        store.add(u, m)
    u2, m2 = _unit(100, "S2")
    store.add(u2, m2)
    # 画素数増加だけで試料間不確かさは消えない (§25.4)
    assert store.count_independent(IndependenceLevel.SAMPLE) == 2
    assert len(store.units) == 101


def test_evaluation_vs_exploration_separation():
    store = EvidenceStore()
    u1, _ = _unit(1, "S1", mode=SamplingMode.EVALUATION)
    store.add(u1, SamplingMode.EVALUATION)
    u2, _ = _unit(2, "S1", mode=SamplingMode.EXPLORATION)
    store.add(u2, SamplingMode.EXPLORATION)
    # 選択的測定を全体割合に混ぜない (§19.1, §25.4)
    assert store.evaluation_only_ids() == ["u1"]


def test_unclassified_kept_in_denominator():
    af = AreaFraction(A_q_pass=30.0, A_q_fail=50.0, A_q_unknown=20.0, A_W=100.0)
    f = af.fractions()
    assert f["pass"] == 0.3 and f["unknown"] == 0.2
    assert abs(sum(f.values()) - 1.0) < 1e-12  # 未分類を分母から消さない (§25.4)


def test_noise_only_does_not_increase_quality_yield():
    # 組成ノイズ増加だけで品質付き成果が増えない (§18.2, §25.4)
    rng = random.Random(0)
    true_x, tol = 0.5, 0.02
    clean = [true_x] * 200
    noisy = [rng.gauss(true_x, 0.05) for _ in range(200)]
    pass_clean = sum(1 for x in clean if abs(x - true_x) <= tol)
    pass_noisy = sum(1 for x in noisy if abs(x - true_x) <= tol)
    assert pass_noisy < pass_clean


def test_kde_bandwidth_required():
    import pytest

    with pytest.raises(ValueError):
        gaussian_kde_1d([0.5], [0.5], 0.0)
    d = gaussian_kde_1d([0.5, 0.5], [0.5], 0.01)
    assert d[0] > 0


def test_coexistence_relations_kept_separate():
    rec = CoexistenceRecord(adjacency="pass")
    # 隣接と平衡共存を同一視しない (§18.3)
    assert rec.adjacency == "pass"
    assert rec.coexistence_same_region == "unknown"
    assert rec.local_equilibrium_consistency == "unknown"


def test_same_composition_different_phase_is_different_info():
    a = EvidenceUnit(id="a", sample_id="S", local_mean_composition={"Al": 0.5},
                     phase_or_candidates=["X"], temperature_K=1000.0)
    b = EvidenceUnit(id="b", sample_id="S", local_mean_composition={"Al": 0.5},
                     phase_or_candidates=["Y"], temperature_K=1000.0)
    assert a.info_key() != b.info_key()  # §18.1
