"""Statistics helpers: percentiles, containment radii, density histograms, chi2, binned tables."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_distribution

from dampe_gamma.binning import IrfBinning, linear_centers

log = logging.getLogger(__name__)

E_BIN_COLUMN = "E_bin"
COS_THETA_BIN_COLUMN = "theta_bin"
"""Bin index columns; the cos(theta) one keeps its historical name used by existing outputs."""


def weighted_percentile(
    data: np.ndarray, weights: np.ndarray, percentiles: float | Sequence[float]
) -> float | np.ndarray:
    """Weighted percentiles, with percentiles given between 0 and 100."""
    data, weights = np.asarray(data), np.asarray(weights)
    percentiles = np.asarray(percentiles, dtype=float)
    if len(data) == 0:
        return np.full(percentiles.shape, np.nan) if percentiles.ndim else np.nan

    sorter = np.argsort(data)
    cumulative_weight = np.cumsum(weights[sorter])
    total_weight = cumulative_weight[-1]
    if total_weight == 0:
        raise ValueError("Sum of weights is zero.")
    return np.interp(percentiles / 100 * total_weight, cumulative_weight, data[sorter])


def containment_radius(values: np.ndarray, percent: float = 68.0) -> float:
    """Radius containing `percent` of the (positive) values."""
    return np.percentile(values, percent)


def weighted_containment_radius(
    values: np.ndarray, weights: np.ndarray, percent: float = 68.0
) -> float:
    """Radius containing `percent` of the total weight."""
    return weighted_percentile(values, weights, percent)


def symmetric_containment_radius(
    data: np.ndarray, weights: np.ndarray | None = None, percent: float = 68.0
) -> float:
    """Smallest radius r such that [-r, r] contains `percent` of the weight."""
    data = np.array(data)
    weights = np.ones(data.shape) if weights is None else np.array(weights)
    if len(data) == 0 or np.all(weights == 0):
        return np.nan

    sorter = np.argsort(np.abs(data))
    data_sorted = data[sorter]
    cumulative_weight = np.cumsum(weights[sorter])
    idx = np.searchsorted(cumulative_weight, percent / 100 * cumulative_weight[-1])
    if idx >= len(data_sorted):
        return np.nan
    return np.abs(data_sorted[idx])


def half_width_68(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Half width of the central 68% interval, between the 16th and 84th percentiles."""
    values = np.asarray(values)
    if weights is None:
        return 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))
    weights = np.asarray(weights, dtype=float)
    if len(values) < 2 or np.all(weights == 0) or np.isnan(weights).all():
        return np.nan
    p16, p84 = weighted_percentile(values, weights, [16, 84])
    return 0.5 * (p84 - p16)


def chisq(
    data: np.ndarray, model: np.ndarray, error: np.ndarray, dofs: int
) -> tuple[float, float, float]:
    """Chi square, reduced chi square and p value of a model against data."""
    chi_squared = np.sum((data - model) ** 2 / error**2)
    p_value = 1 - chi2_distribution.cdf(chi_squared, dofs)
    return chi_squared, chi_squared / dofs, p_value


@dataclass(frozen=True)
class DensityHistogram:
    """Histogram normalised to unit area, with Poisson errors."""

    centers: np.ndarray
    widths: np.ndarray
    counts: np.ndarray
    density: np.ndarray
    errors: np.ndarray
    total: float

    def masked(self, mask: np.ndarray) -> DensityHistogram:
        """Keep only the bins selected by a boolean mask."""
        return DensityHistogram(
            self.centers[mask],
            self.widths[mask],
            self.counts[mask],
            self.density[mask],
            self.errors[mask],
            self.total,
        )


def density_histogram(values: np.ndarray, bins: int | np.ndarray) -> DensityHistogram:
    """Histogram divided by total counts times bin width, with sqrt(N) errors."""
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    total = counts.sum()
    norm = total * widths
    return DensityHistogram(
        linear_centers(edges), widths, counts, counts / norm, np.sqrt(counts) / norm, total
    )


def assign_bins(
    df: pd.DataFrame,
    binning: IrfBinning,
    energy_column: str = "E_true",
    cos_theta_column: str = "cosTheta",
) -> pd.DataFrame:
    """Copy of df with integer energy and cos(theta) bin indices, dropping rows outside the grid."""
    df = df.copy()
    df[E_BIN_COLUMN] = pd.cut(
        df[energy_column], bins=binning.energy, labels=False, include_lowest=True
    )
    df[COS_THETA_BIN_COLUMN] = pd.cut(
        df[cos_theta_column], bins=binning.cos_theta, labels=False, include_lowest=True
    )
    df = df.dropna(subset=[E_BIN_COLUMN, COS_THETA_BIN_COLUMN]).copy()
    df[E_BIN_COLUMN] = df[E_BIN_COLUMN].astype(int)
    df[COS_THETA_BIN_COLUMN] = df[COS_THETA_BIN_COLUMN].astype(int)
    return df


def table_to_grid(table: pd.DataFrame, shape: tuple[int, int], column: str) -> np.ndarray:
    """Per bin table (E_bin, theta_bin, value) to an (energy, cos theta) array, NaN when missing."""
    grid = np.full(shape, np.nan)
    rows = table[E_BIN_COLUMN].to_numpy(dtype=int)
    cols = table[COS_THETA_BIN_COLUMN].to_numpy(dtype=int)
    grid[rows, cols] = table[column].to_numpy()
    return grid


def near_bound_messages(
    popt: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    names: Sequence[str],
    stage: str,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 0.05,
) -> list[str]:
    """Messages for fitted parameters lying on or close to their bounds."""
    messages = []
    for value, lower, upper, name in zip(popt, *bounds, names, strict=True):
        if lower != -np.inf:
            if abs(value - lower) <= abs_tolerance:
                messages.append(f"{stage}: {name} hit its lower bound ({value:.6f}, {lower})")
            elif value > 0 and value <= lower + rel_tolerance * abs(value):
                messages.append(
                    f"{stage}: {name} is close to its lower bound ({value:.6f}, {lower})"
                )
        if upper != np.inf:
            if abs(value - upper) <= abs_tolerance:
                messages.append(f"{stage}: {name} hit its upper bound ({value:.6f}, {upper})")
            elif value >= upper - rel_tolerance * abs(value):
                messages.append(
                    f"{stage}: {name} is close to its upper bound ({value:.6f}, {upper})"
                )
    for message in messages:
        log.debug(message)
    return messages
