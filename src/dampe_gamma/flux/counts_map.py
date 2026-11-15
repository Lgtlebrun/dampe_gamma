"""All sky counts map of the flight photon sample.

As in the original analysis, the energy bin only sets the output folder and the title:
every photon of the sample is counted.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from dampe_gamma import plotting
from dampe_gamma.binning import COARSE_SKY, SkyGrid, geometric_centers
from dampe_gamma.config import Paths
from dampe_gamma.io import load_npz, read_root_columns

log = logging.getLogger(__name__)

TREE = "DmlNtup"
COLUMNS = {
    "celestialcoordBest_2": "lon",
    "celestialcoordBest_3": "lat",
    "tt_bgoTotalEcorr_GeV": "energy",
}


def load_photons(path: str) -> pd.DataFrame:
    """Galactic lon in [-pi, pi], lat (radians) and energy of the flight photons."""
    df = read_root_columns(path, TREE, list(COLUMNS)).rename(columns=COLUMNS)
    df["lon_deg"] = df["lon"] * 180 / np.pi
    df["lat_deg"] = df["lat"] * 180 / np.pi
    df["lon"] = np.where(df["lon"] > np.pi, df["lon"] - 2 * np.pi, df["lon"])
    return df


def counts_map(df: pd.DataFrame, grid: SkyGrid = COARSE_SKY) -> np.ndarray:
    """Photon counts per sky pixel, shape (n_lon, n_lat)."""
    return np.histogram2d(df["lon"], df["lat"], bins=[grid.lon_edges, grid.lat_edges])[0]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--energy-bin", type=int, default=6, help="effective area energy bin used in the labels"
    )


def run(args: argparse.Namespace, paths: Paths) -> None:
    plotting.use_style()
    edges = load_npz(paths.effective_area_file)["E_bins_edge"]
    energy = f"{geometric_centers(edges)[args.energy_bin]:.1f}"
    df = load_photons(str(paths.flight_data_file))
    counts = counts_map(df)

    grid = COARSE_SKY
    fig, _ = plotting.sky_map(
        grid.lon_centers,
        grid.lat_centers,
        counts,
        f"γ-rays Counts, 9 years | Energy : {energy} GeV",
        "Counts",
        log_scale=True,
        facecolor="black",
        lon_label_color="white",
    )
    figure = paths.figures_dir / "fluxMaps" / f"{energy}GeV" / f"CountMapCountMap-{energy}.png"
    plotting.save(fig, figure)
    log.info("Saved counts map to %s", figure)
