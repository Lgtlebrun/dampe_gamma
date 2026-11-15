import numpy as np
import pandas as pd
import pytest

from dampe_gamma import stats
from dampe_gamma.binning import IrfBinning, cos_theta_edges, log_edges


def test_weighted_percentile_unit_weights():
    data = np.arange(1.0, 101.0)
    # cumulative weight convention: the q percentile of 1..100 is q
    assert stats.weighted_percentile(data, np.ones(100), 68) == pytest.approx(68.0)
    assert np.allclose(stats.weighted_percentile(data, np.ones(100), [16, 84]), [16.0, 84.0])


def test_weighted_percentile_follows_weights():
    data, weights = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0])
    assert stats.weighted_percentile(data, weights, 50) == pytest.approx(2.0)
    assert stats.weighted_percentile(data, weights, 75) == pytest.approx(2.5)


def test_weighted_percentile_edge_cases():
    assert np.isnan(stats.weighted_percentile([], [], 68))
    assert np.isnan(stats.weighted_percentile([], [], [16, 84])).all()
    with pytest.raises(ValueError):
        stats.weighted_percentile([1.0, 2.0], [0.0, 0.0], 50)


def test_containment_radii():
    rng = np.random.default_rng(0)
    values = rng.exponential(1.0, 20000)
    assert stats.containment_radius(values) == pytest.approx(np.percentile(values, 68))
    weighted = stats.weighted_containment_radius(values, np.ones_like(values))
    assert weighted == pytest.approx(-np.log(1 - 0.68), rel=0.03)


def test_symmetric_containment_radius_gaussian():
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 2.0, 50000)
    assert stats.symmetric_containment_radius(data) == pytest.approx(2.0 * 0.9945, rel=0.02)
    assert np.isnan(stats.symmetric_containment_radius([]))
    assert np.isnan(stats.symmetric_containment_radius([1.0], [0.0]))


def test_half_width_68_gaussian():
    rng = np.random.default_rng(2)
    data = rng.normal(0.5, 1.5, 50000)
    assert stats.half_width_68(data) == pytest.approx(1.5, rel=0.02)
    assert stats.half_width_68(data, np.ones_like(data)) == pytest.approx(1.5, rel=0.02)
    assert np.isnan(stats.half_width_68([1.0], [1.0]))


def test_chisq():
    data = np.array([1.0, 2.0, 3.0])
    model = np.array([1.0, 1.0, 1.0])
    chi2, chi2_red, p_value = stats.chisq(data, model, np.ones(3), 2)
    assert chi2 == pytest.approx(5.0)
    assert chi2_red == pytest.approx(2.5)
    assert p_value == pytest.approx(np.exp(-2.5))


def test_density_histogram_integrates_to_one():
    rng = np.random.default_rng(3)
    hist = stats.density_histogram(rng.normal(size=1000), 20)
    assert np.sum(hist.density * hist.widths) == pytest.approx(1.0)
    assert np.allclose(hist.errors, np.sqrt(hist.counts) / (hist.total * hist.widths))
    kept = hist.masked(hist.counts > 10)
    assert len(kept.centers) == np.sum(hist.counts > 10)
    assert kept.total == hist.total


def test_assign_bins_and_grid():
    binning = IrfBinning(log_edges(1, 100, 2), cos_theta_edges(2))
    df = pd.DataFrame({"E_true": [1.0, 5.0, 50.0, 200.0], "cosTheta": [0.6, 0.9, 1.0, 0.7]})
    binned = stats.assign_bins(df, binning)
    assert len(binned) == 3
    assert binned[stats.E_BIN_COLUMN].tolist() == [0, 0, 1]
    assert binned[stats.COS_THETA_BIN_COLUMN].tolist() == [0, 1, 1]

    binned["value"] = [1.0, 2.0, 3.0]
    grid = stats.table_to_grid(binned, binning.shape, "value")
    expected = np.array([[1.0, 2.0], [np.nan, 3.0]])
    assert np.array_equal(grid, expected, equal_nan=True)


def test_near_bound_messages():
    bounds = ([0.0, 1.0], [10.0, np.inf])
    assert stats.near_bound_messages([5.0, 3.0], bounds, ["a", "b"], "fit") == []
    messages = stats.near_bound_messages([10.0, 1.0], bounds, ["a", "b"], "fit")
    assert len(messages) == 2
    assert "upper" in messages[0] and "lower" in messages[1]
