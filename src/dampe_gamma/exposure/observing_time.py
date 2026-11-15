"""Observing time of a sky position, histogrammed in detector (cos(theta), phi).

Each housekeeping sample outside the SAA counts once. As in the original analysis,
theta is measured from the nadir axis of the satellite frame.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import GCRS, SkyCoord

from dampe_gamma import plotting
from dampe_gamma.binning import cos_theta_edges, phi_edges_deg
from dampe_gamma.config import Paths
from dampe_gamma.exposure.detector_frame import frame_from_hk, track_angles
from dampe_gamma.exposure.housekeeping import (
    date_from_path,
    hk_times,
    month_files,
    parse_date,
    read_hk,
    restrict_to_day,
)
from dampe_gamma.exposure.saa import mask_saa
from dampe_gamma.io import ensure_dir

log = logging.getLogger(__name__)

COS_THETA_EDGES = cos_theta_edges(10)
PHI_EDGES = phi_edges_deg(360)


def target_angles(df: pd.DataFrame, target: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """cos(theta) from the nadir and azimuth in degrees of a target at each sample."""
    target_gcrs = target.transform_to(GCRS(obstime=hk_times(df)))
    directions = target_gcrs.cartesian.get_xyz().value.T
    return track_angles(frame_from_hk(df, robust=False), directions, zenith=False)


def day_histogram(file: str, date: datetime.date, target: SkyCoord) -> np.ndarray:
    """Histogram of one housekeeping file restricted to its UTC day."""
    df = mask_saa(restrict_to_day(read_hk([file]), date))[0]
    cos_theta, phi = target_angles(df, target)
    return np.histogram2d(cos_theta, phi, bins=[COS_THETA_EDGES, PHI_EDGES])[0]


def months(start: datetime.date, end: datetime.date) -> Iterator[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def accumulate(
    hk_dir: Path, target: SkyCoord, start: datetime.date, end: datetime.date
) -> np.ndarray:
    """Observing time histogram summed over the days between start and end included."""
    total = np.zeros((len(COS_THETA_EDGES) - 1, len(PHI_EDGES) - 1))
    for year, month in months(start, end):
        files = month_files(hk_dir, year, month)
        if not files:
            log.warning("No files found for %d-%02d", year, month)
        for file in files:
            date = date_from_path(file)
            if not start <= date <= end:
                continue
            log.info("Processing %s", file)
            total += day_histogram(file, date, target)
    return total


def plot_histogram2d(histogram: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    phi_grid, cos_theta_grid = np.meshgrid(PHI_EDGES, COS_THETA_EDGES)
    mesh = ax.pcolormesh(phi_grid, cos_theta_grid, histogram, shading="auto", cmap="viridis")
    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\cos(\theta)$")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Observing Time [s]")
    fig.tight_layout()
    plotting.save(fig, path)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("l", type=float, help="galactic longitude in degrees")
    parser.add_argument("b", type=float, help="galactic latitude in degrees")
    parser.add_argument("--name", required=True, help="target name used in output paths")
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--month", help="one month, YYYY-MM")
    period.add_argument("--start", type=parse_date, help="first day, YYYY-MM-DD (needs --end)")
    parser.add_argument("--end", type=parse_date, help="last day included, YYYY-MM-DD")


def run(args: argparse.Namespace, paths: Paths) -> None:
    plotting.use_style()
    target = SkyCoord(l=args.l, b=args.b, unit="deg", frame="galactic")
    log.info("Observing time of %s at l=%s, b=%s", args.name, args.l, args.b)
    result_root = paths.results_dir / "obsTime"
    figure_root = paths.figures_dir / "obsTime"

    if args.month is not None:
        year, month = (int(x) for x in args.month.split("-"))
        start = datetime.date(year, month, 1)
        end = datetime.date(year + month // 12, month % 12 + 1, 1) - datetime.timedelta(days=1)
        histogram = accumulate(paths.hk_dir, target, start, end)
        stem = Path(args.name) / f"{year}" / f"{month:02d}"
        result = ensure_dir(result_root / stem.parent) / f"{month:02d}_histogram.npz"
        figure = figure_root / stem.parent / f"{month:02d}_histogram2D.png"
        title = "2D Observation Distribution"
    else:
        if args.end is None:
            raise SystemExit("--start needs --end")
        histogram = accumulate(paths.hk_dir, target, args.start, args.end)
        if not histogram.any():
            raise SystemExit("No data accumulated over the date range.")
        name = f"{args.name}_{args.start}_{args.end}"
        result = ensure_dir(result_root) / f"{name}_histogram.npz"
        figure = figure_root / f"{name}_histogram2D.png"
        title = f"2D Observation Distribution ({args.start} to {args.end})"

    np.savez(result, histogram=histogram)
    plot_histogram2d(histogram, title, figure)
    log.info("Saved %s and %s", result, figure)
