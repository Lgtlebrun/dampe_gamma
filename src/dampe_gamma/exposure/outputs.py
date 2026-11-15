"""Energy bins, output locations and figures shared by the exposure commands."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dampe_gamma import plotting
from dampe_gamma.binning import geometric_centers, giem_energy_edges
from dampe_gamma.config import Paths
from dampe_gamma.io import load_npz

log = logging.getLogger(__name__)

EXPOSURE_LABEL = "Exposure [cm² s]"


def energy_bin_centers(paths: Paths) -> np.ndarray:
    """GIEM energy centres in GeV, read from the saved bins file when it exists."""
    if paths.giem_bins_file.is_file():
        edges = load_npz(paths.giem_bins_file)["E_bins_edge"]
    else:
        log.info("%s not found, using the built in GIEM edges", paths.giem_bins_file)
        edges = giem_energy_edges()
    return geometric_centers(edges)


def par_dir(paths: Paths) -> Path:
    return paths.data_dir / "par"


def all_sky_dir(paths: Paths, energy: float) -> Path:
    return paths.work_dir / "exposureMaps_GIE" / f"{energy:.3f}GeV"


def target_dir(paths: Paths, target: str, energy: float) -> Path:
    return paths.work_dir / "exposureMaps_targets" / target / f"{energy:.3f}GeV"


def target_figure_dir(paths: Paths, target: str, energy: float) -> Path:
    return paths.figures_dir / target / "exposureMaps" / f"{energy:.3f}GeV"


def plot_roi(
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    exposure: np.ndarray,
    target_lon: float,
    target_lat: float,
    target: str,
    energy: float,
    path: Path,
) -> None:
    """Save the exposure map of a target region."""
    fig, _ = plotting.roi_map(
        lon_centers,
        lat_centers,
        exposure,
        target_lon,
        target_lat,
        target,
        f"Exposure Map of {target} ROI at {energy:.3f} GeV",
        EXPOSURE_LABEL,
    )
    plotting.save(fig, path)
