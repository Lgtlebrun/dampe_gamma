"""Sum daily exposure maps, all sky or around a target, for one energy bin."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dampe_gamma import plotting
from dampe_gamma.binning import linear_centers
from dampe_gamma.config import Paths
from dampe_gamma.ephemeris import target_galactic_coordinates
from dampe_gamma.exposure import outputs
from dampe_gamma.io import browse_files, ensure_dir

log = logging.getLogger(__name__)


def daily_files(directory: str | Path) -> list[str]:
    return browse_files(Path(directory) / "*.npz")


def read_edges(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["lon_edges"], data["lat_edges"]


def sum_daily_maps(files: Sequence[str], skip_unreadable: bool = False) -> np.ndarray | None:
    """Sum of the exposure_map arrays of several npz files, read one at a time."""
    summed = None
    for i, path in enumerate(files):
        try:
            with np.load(path) as data:
                exposure = data["exposure_map"]
        except Exception:
            if not skip_unreadable:
                raise
            log.warning("Could not read %s", path, exc_info=True)
            continue
        if summed is None:
            summed = np.zeros_like(exposure)
        summed += exposure
        if i % 50 == 0:
            log.info("Summed %d/%d maps", i + 1, len(files))
    return summed


def flip_longitude(lon: np.ndarray) -> np.ndarray:
    """Map longitudes to [-pi, pi] and reverse their sign, for east to the left plots."""
    return -np.where(lon > np.pi, lon - 2 * np.pi, lon)


def sum_all_sky(paths: Paths, energy: float) -> np.ndarray:
    """Sum the daily all sky maps of one energy and save the total twice, with flipped lon edges."""
    data_dir = outputs.all_sky_dir(paths, energy)
    files = daily_files(data_dir / "daily")
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir / 'daily'}")
    log.info("Found %d daily maps in %s", len(files), data_dir / "daily")

    lon_edges, lat_edges = read_edges(files[0])
    lon_centers, lat_centers = linear_centers(lon_edges), linear_centers(lat_edges)
    lon_edges_flipped = flip_longitude(lon_edges)
    exposure = sum_daily_maps(files, skip_unreadable=True)

    fig, _ = plotting.sky_map(
        flip_longitude(lon_centers),
        lat_centers,
        exposure,
        f"Exposure Map ({energy:.3f} GeV)",
        outputs.EXPOSURE_LABEL,
        lon_label_color="white",
    )
    plotting.save(fig, paths.figures_dir / "exposureMaps_GIE" / f"{energy:.3f}GeV_exposure_sum.png")

    name = f"{energy:.3f}GeV_exposure_sum.npz"
    for path in (ensure_dir(paths.results_dir / "exposureMaps") / name, data_dir / name):
        np.savez(path, lon_edges=lon_edges_flipped, lat_edges=lat_edges, exposure_map=exposure)
        log.info("Saved %s", path)
    return exposure


def sum_target(
    paths: Paths,
    target: str,
    energy: float,
    target_lon: float,
    target_lat: float,
    suffix: str = "",
) -> np.ndarray | None:
    """Sum every daily map of a target found in its daily folder, then save and plot it."""
    result_dir = outputs.target_dir(paths, target, energy)
    files = daily_files(result_dir / "daily")
    if not files:
        log.warning("No exposure maps found to sum in %s", result_dir / "daily")
        return None
    log.info("Found %d daily maps in %s", len(files), result_dir / "daily")

    lon_edges, lat_edges = read_edges(files[0])
    exposure = sum_daily_maps(files)
    output = result_dir / f"summed_exposure_map{suffix}.npz"
    np.savez_compressed(output, exposure_map=exposure, lon_edges=lon_edges, lat_edges=lat_edges)
    log.info("Summed %d maps into %s", len(files), output)

    figure = outputs.target_figure_dir(paths, target, energy) / f"summed_exposure_map{suffix}.png"
    outputs.plot_roi(
        linear_centers(lon_edges),
        linear_centers(lat_edges),
        exposure,
        target_lon,
        target_lat,
        target,
        energy,
        figure,
    )
    return exposure


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--energy-bin", type=int, required=True, help="index of the GIEM energy bin"
    )
    parser.add_argument("--target", help="sum the maps of this target instead of the all sky maps")


def run(args: argparse.Namespace, paths: Paths) -> None:
    plotting.use_style()
    energy = outputs.energy_bin_centers(paths)[args.energy_bin]
    if args.target is None:
        sum_all_sky(paths, energy)
        return
    lon, lat = target_galactic_coordinates(args.target, outputs.par_dir(paths))
    sum_target(paths, args.target, energy, lon, lat)
