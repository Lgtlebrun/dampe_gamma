import numpy as np
import pandas as pd
import pytest

from dampe_gamma.binning import IrfBinning, cos_theta_edges, energy_edges, phi_edges_deg
from dampe_gamma.irf import effective_area as ea


@pytest.fixture
def binning():
    return IrfBinning(energy_edges("log", 4), cos_theta_edges(2), phi_edges_deg(4))


def make_events(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tt_ekin": 10 ** rng.uniform(3, 7, n),
            "tt_theta": np.arccos(rng.uniform(0.5, 1, n)),
            "tt_phi": rng.uniform(-np.pi, np.pi, n),
            "weight": rng.uniform(0.5, 1.5, n),
        }
    )


def test_preprocess_adds_derived_columns():
    df = ea.preprocess(make_events(100))
    assert np.allclose(df["E_true"], df["tt_ekin"] / 1000)
    assert df["cosTheta"].between(0.5, 1).all()
    assert df["phi_True_deg"].between(0, 360).all()


def test_efficiency_of_random_subsample(binning):
    total = ea.preprocess(make_events(200_000))
    selected = total.sample(frac=0.3, random_state=1)
    eff = ea.compute_efficiency(total, selected, binning)
    assert eff.full.shape == binning.shape
    assert eff.energy_cos_theta.shape == binning.shape[:2]
    assert np.allclose(eff.energy, 0.3, atol=0.01)
    assert np.allclose(eff.energy_weighted, 0.3, atol=0.01)
    assert np.all(eff.energy_error > 0)


def test_empty_bins_give_zero_efficiency(binning):
    total = ea.preprocess(make_events(10))
    eff = ea.compute_efficiency(total, total.iloc[:0], binning)
    assert np.all(eff.full == 0)


def test_save_and_load_round_trip(tmp_path, binning):
    table = np.random.default_rng(0).random(binning.shape)
    ea.save_effective_area(tmp_path / "a.npz", table, binning)
    loaded, loaded_binning = ea.load_effective_area(tmp_path / "a.npz")
    assert np.array_equal(loaded, table)
    assert np.array_equal(loaded_binning.phi_deg, binning.phi_deg)
