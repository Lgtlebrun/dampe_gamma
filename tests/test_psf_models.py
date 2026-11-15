import numpy as np
import pytest
from scipy.integrate import quad

from dampe_gamma.irf import psf_models as pm


@pytest.mark.parametrize(("sigma", "gamma"), [(0.5, 1.6), (1.0, 2.5), (2.0, 10.0)])
def test_king_is_normalised_over_the_plane(sigma, gamma):
    integral, _ = quad(lambda x: 2 * np.pi * x * pm.king(x, sigma, gamma), 0, np.inf)
    assert integral == pytest.approx(1.0, rel=1e-6)


def test_radial_king_integrates_to_amplitude():
    integral, _ = quad(lambda x: pm.radial_king(x, 0.7, 0.8, 3.0), 0, np.inf)
    assert integral == pytest.approx(0.7, rel=1e-6)


def test_double_king_is_core_plus_tail():
    x = np.logspace(-2, 1.5, 200)
    A, F, sigma_c, gamma_c, sigma_t, gamma_t = 1.3, 0.75, 0.5, 2.5, 2.0, 1.8
    total = pm.double_king(x, A, F, sigma_c, gamma_c, sigma_t, gamma_t)
    core = pm.core_model(x, A * F, sigma_c, gamma_c)
    tail = pm.tail_model(x, A * (1 - F), sigma_t, gamma_t)
    assert np.allclose(total, core + tail, rtol=1e-12)
    integral, _ = quad(pm.double_king, 0, np.inf, args=(A, F, sigma_c, gamma_c, sigma_t, gamma_t))
    assert integral == pytest.approx(A, rel=1e-6)


def test_double_king_from_amplitudes():
    x = np.linspace(0.01, 5, 50)
    expected = pm.double_king(x, 1.0, 0.6, 0.5, 2.5, 2.0, 1.8)
    assert np.allclose(pm.double_king_from_amplitudes(x, 0.6, 0.5, 2.5, 0.4, 2.0, 1.8), expected)
