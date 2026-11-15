import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from dampe_gamma.binning import IrfBinning, cos_theta_edges, log_edges
from dampe_gamma.irf import energy_dispersion as ed
from dampe_gamma.stats import assign_bins

D_PARAMS = (0.0, -2.0, 0.5, 2.0, 1.5, 0.6, 2.0, 0.4, 1.5, 0.9)


@pytest.mark.parametrize(("sigma", "k", "b", "p"), [(1.0, 1.0, 0.0, 2.0), (0.5, 1.7, -0.3, 1.3)])
def test_asymmetric_gaussian_is_normalised(sigma, k, b, p):
    left, _ = quad(ed.asymmetric_gaussian, -np.inf, b, args=(sigma, k, b, p))
    right, _ = quad(ed.asymmetric_gaussian, b, np.inf, args=(sigma, k, b, p))
    assert left + right == pytest.approx(1.0, rel=1e-6)


def test_dispersion_function_core_and_continuity_at_x0_zero():
    x = np.array([D_PARAMS[1] - 1e-9, D_PARAMS[1] + 1e-9, 0.0, D_PARAMS[2] + 1e-9])
    values = ed.dispersion_function(x, *D_PARAMS)
    assert values[2] == pytest.approx(D_PARAMS[-1])
    assert values[0] == pytest.approx(values[1], rel=1e-6)
    inner = D_PARAMS[-1] * ed.exp_power(D_PARAMS[2], 0.0, D_PARAMS[5], D_PARAMS[6])
    assert values[3] == pytest.approx(inner, rel=1e-6)


def test_preprocess_computes_relative_deviation():
    df = pd.DataFrame(
        {
            "tt_theta": [0.2, 0.2],
            "tt_ekin": [10e3, 2e7],
            "tt_bgoTotalEcorr_GeV": [11.0, 2e4],
            "weight": [1.0, 1.0],
        }
    )
    out = ed.preprocess(df)
    assert len(out) == 1
    assert out["Ediff"].iloc[0] == pytest.approx(0.1)
    assert out["cosTheta"].iloc[0] == pytest.approx(np.cos(0.2))


def test_compute_resolution_columns():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame(
        {
            "E_true": 10 ** rng.uniform(0, 2, n),
            "cosTheta": rng.uniform(0.5, 1, n),
            "Ediff": rng.normal(0, 0.1, n),
            "weight": np.ones(n),
        }
    )
    binning = IrfBinning(log_edges(1, 100, 2), cos_theta_edges(2))
    table = ed.compute_resolution(assign_bins(df, binning))
    assert list(table.columns) == ["E_bin", "theta_bin", *ed.RESOLUTION_COLUMNS]
    assert len(table) == 4
    assert np.allclose(table[list(ed.RESOLUTION_COLUMNS)], 0.1, rtol=0.15)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fit_xd_bin_runs_and_reports_empty_bins():
    rng = np.random.default_rng(1)
    x_d = np.concatenate([rng.normal(0, 1, 5000), rng.normal(-3, 2, 1000)])
    fit = ed.fit_xd_bin(x_d, n_starts=1, rng=np.random.default_rng(1))
    assert fit.params.shape == (len(ed.D_PARAM_NAMES),)
    assert fit.ndof == len(fit.histogram.centers) - len(ed.D_PARAM_NAMES)
    assert np.isfinite(fit.chi2)
    with pytest.raises(ed.EmptyBinError):
        ed.fit_xd_bin(np.array([]))
