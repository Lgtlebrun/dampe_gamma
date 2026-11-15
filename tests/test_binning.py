import numpy as np
import pytest

from dampe_gamma import binning


def test_log_edges_span_and_count():
    edges = binning.log_edges(1, 1e4, 40)
    assert len(edges) == 41
    assert edges[0] == pytest.approx(1)
    assert edges[-1] == pytest.approx(1e4)
    assert np.allclose(np.diff(np.log10(edges)), 0.1)


def test_edges_from_log_centers_round_trip():
    edges = binning.log_edges(1, 100, 8)
    centers = binning.geometric_centers(edges)
    assert np.allclose(binning.edges_from_log_centers(centers), edges)


def test_giem_edges_extend_to_twenty_bins():
    edges = binning.giem_energy_edges()
    giem = binning.GIEM_ENERGY_CENTERS
    assert len(edges) == 21
    assert np.all(np.diff(edges) > 0)
    assert np.all((edges[:16] < giem) & (giem < edges[1:17]))


def test_energy_edges_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        binning.energy_edges("linear")


def test_irf_binning_shape():
    b = binning.IrfBinning(
        binning.energy_edges("log", 40), binning.cos_theta_edges(10), binning.phi_edges_deg()
    )
    assert b.shape == (40, 10, 360)
    assert b.cos_theta_centers[0] == pytest.approx(0.525)


def test_sky_grid_reproduces_original_latitude_recipe():
    grid = binning.COARSE_SKY
    lat_centers = np.linspace(-np.pi / 2, np.pi / 2, 361)
    expected = np.zeros(362)
    expected[0] = lat_centers[0] - (lat_centers[1] - lat_centers[0])
    expected[1:] = lat_centers
    assert np.array_equal(grid.lat_edges, expected)
    assert len(grid.lon_centers) == 720
