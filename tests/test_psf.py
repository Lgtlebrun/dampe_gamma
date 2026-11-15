import numpy as np
import pandas as pd
import pytest

from dampe_gamma.binning import IrfBinning, cos_theta_edges, log_edges
from dampe_gamma.irf import psf
from dampe_gamma.irf.psf_models import double_king


def sample_king(n, sigma, gamma, rng):
    """Inverse transform sampling of the radial King distribution 2 pi x P(x)."""
    u = rng.uniform(size=n)
    return sigma * np.sqrt(2 * gamma * ((1 - u) ** (1 / (1 - gamma)) - 1))


def test_angular_error_simple_cases():
    zero = np.zeros(1)
    assert psf.angular_error(zero, zero, zero, zero) == pytest.approx(0.0)
    half_pi = np.full(1, np.pi / 2)
    assert psf.angular_error(zero, zero, half_pi, zero) == pytest.approx(np.pi / 2)
    assert psf.angular_error(half_pi, zero, half_pi, half_pi) == pytest.approx(np.pi / 2)


def test_preprocess_and_angular_error():
    df = pd.DataFrame(
        {
            "tt_ekin": [5e3, 2e7, 1e5],
            "tt_theta": [0.3, 0.3, 0.0],
            "tt_phi": [0.1, 0.1, 0.1],
            "theta_BestSTK": [np.rad2deg(0.31), 10.0, 10.0],
            "phi_BestSTK": [np.rad2deg(0.1), 0.0, 0.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    out = psf.compute_angular_error(psf.preprocess(df))
    assert len(out) == 1
    assert out["E_true"].iloc[0] == pytest.approx(5.0)
    assert out["angular_error_degrees"].iloc[0] == pytest.approx(np.rad2deg(0.01))


def test_compute_resolution_on_synthetic_events():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(
        {
            "E_true": 10 ** rng.uniform(0, 2, n),
            "cosTheta": rng.uniform(0.5, 1, n),
            "angular_error_degrees": rng.exponential(1.0, n),
            "weight": rng.uniform(0.5, 2, n),
        }
    )
    binning = IrfBinning(log_edges(1, 100, 2), cos_theta_edges(2))
    merged, resolution, resolution_w = psf.compute_resolution(df, binning)

    assert list(resolution.columns) == ["E_bin", "theta_bin", "angular_resolution"]
    assert len(resolution) == len(resolution_w) == 4
    for row in resolution.itertuples():
        in_bin = (merged["E_bin"] == row.E_bin) & (merged["theta_bin"] == row.theta_bin)
        expected = np.percentile(merged.loc[in_bin, "angular_error_degrees"], 68)
        assert row.angular_resolution == pytest.approx(expected)
    assert merged["angular_resolution"].notna().all()
    assert np.allclose(resolution_w["angular_resolution"], -np.log(0.32), rtol=0.15)


def test_two_stage_fit_recovers_double_king():
    rng = np.random.default_rng(0)
    sigma_c, gamma_c, sigma_t, gamma_t, F, n = 0.5, 3.0, 2.0, 2.0, 0.8, 20000
    n_core = int(F * n)
    xp = np.concatenate(
        [sample_king(n_core, sigma_c, gamma_c, rng), sample_king(n - n_core, sigma_t, gamma_t, rng)]
    )
    fit = psf.fit_bin_two_stage(xp, psf.xp_bins(xp))

    assert fit.params["A"] == pytest.approx(1.0, rel=0.05)
    assert fit.params["F"] == pytest.approx(F, abs=0.1)
    assert fit.params["sigma_c"] == pytest.approx(sigma_c, rel=0.1)
    assert fit.params["sigma_t"] == pytest.approx(sigma_t, rel=0.35)
    assert fit.chi2_red < psf.FALLBACK_CHI2_RED
    assert np.allclose(fit.model_total, fit.core_fit + fit.tail_fit)
    assert np.all(fit.histogram.centers > psf.MIN_XP_CENTER)


def test_full_fit_on_exact_model():
    x = np.logspace(-1.5, 1, 25)
    truth = (0.8, 0.5, 2.5, 0.2, 2.0, 2.0)
    A = truth[0] + truth[3]
    y = double_king(x, A, truth[0] / A, truth[1], truth[2], truth[4], truth[5])
    fit = psf.full_fit(x, y, 0.01 * y)
    assert fit.method == "full"
    assert fit.params["A"] == pytest.approx(A, rel=1e-3)
    assert fit.params["sigma_c"] == pytest.approx(0.5, rel=1e-3)


def test_xp_bins_count():
    assert len(psf.xp_bins(np.linspace(0.1, 1, 20))) == psf.MIN_XP_BINS
    assert len(psf.xp_bins(np.linspace(0.1, 1, 10000))) == psf.MAX_XP_BINS


def test_default_binning():
    binning = psf.default_binning()
    assert binning.shape == (8, 5)
    assert binning.energy[0] == pytest.approx(1.0)
    assert binning.energy[-1] == pytest.approx(1e4)
    assert psf.default_binning("giem").shape[1] == 5
