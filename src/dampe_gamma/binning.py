"""Binning schemes shared by the selection, the IRFs and the exposure maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

E_MIN_GEV = 1.0
E_MAX_GEV = 1e4

GIEM_ENERGY_CENTERS = np.array(
    [
        1.16121705e00,
        1.50918340e00,
        1.96142017e00,
        2.54917267e00,
        3.31304908e00,
        4.30582611e00,
        7.27300221e00,
        1.22848810e01,
        2.07504819e01,
        3.50497899e01,
        5.92028552e01,
        9.99999737e01,
        1.68910683e02,
        2.85308264e02,
        4.81916266e02,
        8.14008272e02,
    ]
)
"""Energy bin centres (GeV) of the Fermi Galactic Interstellar Emission Model."""


def log_edges(e_min: float, e_max: float, n_bins: int) -> np.ndarray:
    """Logarithmically spaced bin edges."""
    return np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)


def geometric_centers(edges: np.ndarray) -> np.ndarray:
    """Geometric mean of consecutive edges, the natural centre of log bins."""
    edges = np.asarray(edges)
    return np.sqrt(edges[:-1] * edges[1:])


def linear_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges)
    return 0.5 * (edges[:-1] + edges[1:])


def edges_from_log_centers(centers: np.ndarray) -> np.ndarray:
    """Bin edges around log spaced centres, extrapolating the outer edges."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] / 2, centers[0] * 2])
    first = centers[0] / np.sqrt(centers[1] / centers[0])
    last = centers[-1] * np.sqrt(centers[-1] / centers[-2])
    return np.concatenate([[first], geometric_centers(centers), [last]])


def giem_energy_edges(n_extra_bins: int = 4) -> np.ndarray:
    """GIEM energy edges, extended to higher energies with the coarsest GIEM ratio."""
    ratio = GIEM_ENERGY_CENTERS[-1] / GIEM_ENERGY_CENTERS[-2]
    extra = GIEM_ENERGY_CENTERS[-1] * ratio ** np.arange(1, n_extra_bins + 1)
    return edges_from_log_centers(np.concatenate([GIEM_ENERGY_CENTERS, extra]))


def cos_theta_edges(n_bins: int, cos_min: float = 0.5, cos_max: float = 1.0) -> np.ndarray:
    return np.linspace(cos_min, cos_max, n_bins + 1)


def phi_edges_deg(n_bins: int = 360) -> np.ndarray:
    return np.linspace(0.0, 360.0, n_bins + 1)


def energy_edges(scheme: str, n_bins: int = 40) -> np.ndarray:
    """Energy edges in GeV for a named scheme: 'log' (n_bins between 1 GeV and 10 TeV) or 'giem'."""
    if scheme == "log":
        return log_edges(E_MIN_GEV, E_MAX_GEV, n_bins)
    if scheme == "giem":
        return giem_energy_edges()
    raise ValueError(f"Unknown energy binning scheme: {scheme!r}")


@dataclass(frozen=True)
class IrfBinning:
    """Energy, cos(theta) and phi edges on which an IRF is tabulated."""

    energy: np.ndarray
    cos_theta: np.ndarray
    phi_deg: np.ndarray | None = None

    @property
    def energy_centers(self) -> np.ndarray:
        return geometric_centers(self.energy)

    @property
    def cos_theta_centers(self) -> np.ndarray:
        return linear_centers(self.cos_theta)

    @property
    def phi_centers(self) -> np.ndarray:
        return linear_centers(self.phi_deg)

    @property
    def shape(self) -> tuple[int, ...]:
        axes = [self.energy, self.cos_theta] + ([self.phi_deg] if self.phi_deg is not None else [])
        return tuple(len(a) - 1 for a in axes)


@dataclass(frozen=True)
class SkyGrid:
    """Galactic longitude and latitude grid, in radians.

    Latitude edges follow the original recipe: the first edge is one step below the
    first centre and the remaining edges coincide with the centres.
    """

    n_lon: int
    n_lat: int

    @property
    def lon_edges(self) -> np.ndarray:
        return np.linspace(-np.pi, np.pi, self.n_lon + 1)

    @property
    def lon_centers(self) -> np.ndarray:
        return linear_centers(self.lon_edges)

    @property
    def lat_centers(self) -> np.ndarray:
        return np.linspace(-np.pi / 2, np.pi / 2, self.n_lat)

    @property
    def lat_edges(self) -> np.ndarray:
        centers = self.lat_centers
        return np.concatenate([[centers[0] - (centers[1] - centers[0])], centers])

    @property
    def dlon(self) -> float:
        return float(self.lon_edges[1] - self.lon_edges[0])

    @property
    def dlat(self) -> float:
        return float(self.lat_centers[1] - self.lat_centers[0])


COARSE_SKY = SkyGrid(720, 361)
FINE_SKY = SkyGrid(2880, 1441)
