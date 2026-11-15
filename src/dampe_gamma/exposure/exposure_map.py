"""Daily exposure maps, over the whole sky or in a square region around a target.

Each housekeeping sample outside the SAA adds, to every sky pixel within the field of
view, the effective area at the pixel's detector angles. Samples are counted, not
weighted by their duration, as in the original analysis.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from dampe_gamma import plotting
from dampe_gamma.binning import COARSE_SKY, FINE_SKY, SkyGrid
from dampe_gamma.config import Paths
from dampe_gamma.ephemeris import target_galactic_coordinates
from dampe_gamma.exposure import outputs
from dampe_gamma.exposure.detector_frame import (
    SatelliteFrame,
    frame_from_hk,
    grid_cos_theta,
    grid_phi_deg,
)
from dampe_gamma.exposure.housekeeping import (
    available_dates,
    hk_times,
    load_day,
    parse_date,
)
from dampe_gamma.exposure.saa import mask_saa
from dampe_gamma.exposure.sum_maps import sum_target
from dampe_gamma.io import ensure_dir

log = logging.getLogger(__name__)

FOV_DEG = 60
BATCH_SIZE = 1000
FIRST_YEAR, LAST_YEAR = 2016, 2024
DEFAULT_ROI_DEG = {"year": 4.0, "range": 6.0}


@dataclass(frozen=True)
class EffectiveAreaTable:
    """Effective area tabulated in (E, cos(theta), phi) with its bin edges."""

    values: np.ndarray
    energy_edges: np.ndarray
    cos_theta_edges: np.ndarray
    phi_edges: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> EffectiveAreaTable:
        from dampe_gamma.irf.effective_area import load_effective_area

        values, binning = load_effective_area(path)
        return cls(values, binning.energy, binning.cos_theta, binning.phi_deg)

    def lookup(self, cos_theta: np.ndarray, phi_deg: np.ndarray, energy: float) -> np.ndarray:
        """Effective area of the bin holding each point, zero outside the table."""
        cos_theta = np.asarray(cos_theta)
        energy_idx = np.digitize(energy, self.energy_edges) - 1
        cos_theta_idx = np.digitize(cos_theta, self.cos_theta_edges) - 1
        phi_idx = np.digitize(phi_deg, self.phi_edges) - 1
        shape = self.values.shape
        valid = (
            (energy_idx >= 0)
            & (energy_idx < shape[0])
            & (cos_theta_idx >= 0)
            & (cos_theta_idx < shape[1])
            & (phi_idx >= 0)
            & (phi_idx < shape[2])
        )
        area = np.zeros_like(cos_theta, dtype=float)
        if valid.any():
            area[valid] = self.values[energy_idx, cos_theta_idx[valid], phi_idx[valid]]
        return area


def galactic_to_icrs_vectors(lon_rad: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
    """ICRS unit vectors, shape (M, 3), of galactic directions given in radians."""
    coords = SkyCoord(lon_rad.ravel() * u.rad, lat_rad.ravel() * u.rad, frame="galactic")
    return coords.transform_to("icrs").cartesian.xyz.value.T


def accumulate_exposure(
    frame: SatelliteFrame,
    directions: np.ndarray,
    area: EffectiveAreaTable,
    energy: float,
    fov_deg: float = FOV_DEG,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Sum over samples of the effective area seen in each direction, shape (M,)."""
    exposure = np.zeros(directions.shape[0], dtype=np.float64)
    for start in range(0, len(frame), batch_size):
        batch = frame[start : start + batch_size]
        cos_theta = grid_cos_theta(batch, directions, zenith=True)
        in_fov = cos_theta >= np.cos(np.radians(fov_deg))
        if not np.any(in_fov):
            continue
        phi = grid_phi_deg(batch, directions)
        increments = np.zeros_like(cos_theta, dtype=np.float32)
        increments[in_fov] = area.lookup(cos_theta[in_fov], phi[in_fov], energy)
        exposure += increments.sum(axis=0)
        log.debug("Processed samples %d to %d", start, start + len(batch) - 1)
    return exposure


def compute_exposure_map(
    df: pd.DataFrame, area: EffectiveAreaTable, energy: float, grid: SkyGrid = COARSE_SKY
) -> np.ndarray:
    """All sky exposure, shape (n_lon, n_lat), of one day of housekeeping samples."""
    lon_grid, lat_grid = np.meshgrid(grid.lon_centers, grid.lat_centers, indexing="ij")
    directions = galactic_to_icrs_vectors(lon_grid, lat_grid)
    exposure = accumulate_exposure(frame_from_hk(df), directions, area, energy)
    return exposure.reshape(lon_grid.shape)


def roi_indices(
    grid: SkyGrid, lon_deg: float, lat_deg: float, side_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel indices of an odd sized square around a position, wrapped in lon, clipped in lat."""
    n_lon = int(np.round(np.deg2rad(side_deg) / grid.dlon))
    n_lat = int(np.round(np.deg2rad(side_deg) / grid.dlat))
    if n_lon % 2 == 0:
        n_lon += 1
    if n_lat % 2 == 0:
        n_lat += 1
    center_lon = np.argmin(np.abs(grid.lon_centers - np.deg2rad(lon_deg)))
    center_lat = np.argmin(np.abs(grid.lat_centers - np.deg2rad(lat_deg)))
    lon_idx = np.arange(center_lon - n_lon // 2, center_lon + n_lon // 2 + 1) % grid.n_lon
    lat_idx = np.arange(center_lat - n_lat // 2, center_lat + n_lat // 2 + 1)
    return lon_idx, np.clip(lat_idx, 0, grid.n_lat - 1)


def compute_exposure_map_roi(
    df: pd.DataFrame,
    area: EffectiveAreaTable,
    energy: float,
    lon_deg: float,
    lat_deg: float,
    side_deg: float = DEFAULT_ROI_DEG["year"],
    grid: SkyGrid = FINE_SKY,
) -> dict[str, np.ndarray]:
    """Exposure in a square region of interest and its grid."""
    lon_idx, lat_idx = roi_indices(grid, lon_deg, lat_deg, side_deg)
    lon_centers, lat_centers = grid.lon_centers[lon_idx], grid.lat_centers[lat_idx]
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing="ij")
    directions = galactic_to_icrs_vectors(lon_grid, lat_grid)
    exposure = accumulate_exposure(frame_from_hk(df), directions, area, energy)
    return {
        "exposure": exposure.reshape(lon_grid.shape),
        "lon_indices": lon_idx,
        "lat_indices": lat_idx,
        "lon_grid": lon_grid,
        "lat_grid": lat_grid,
        "lon_centers": lon_centers,
        "lat_centers": lat_centers,
        "lon_edges": grid.lon_edges[lon_idx[0] : lon_idx[-1] + 2],
        "lat_edges": grid.lat_edges[lat_idx[0] : lat_idx[-1] + 2],
    }


def load_day_outside_saa(paths: Paths, date: datetime.date) -> pd.DataFrame:
    df = load_day(paths.hk_dir, date)
    log.info("Loaded %d samples for %s", len(df), date)
    if len(df):
        utc = hk_times(df.iloc[[0, -1]])
        log.info("Start time: %s, end time: %s", utc[0].iso, utc[1].iso)
    df, valid = mask_saa(df)
    log.info("Masked out %d samples inside the SAA", np.sum(~valid))
    return df


def run_all_sky(paths: Paths, date: datetime.date, area: EffectiveAreaTable, energy: float) -> None:
    """Exposure map of the whole sky for one day."""
    df = load_day_outside_saa(paths, date)
    if len(df) == 0:
        log.warning("No data found for %s", date)
        return
    grid = COARSE_SKY
    exposure = compute_exposure_map(df, area, energy, grid)

    figure = (
        paths.figures_dir
        / "exposureMaps_GIE"
        / f"{energy:.3f}GeV"
        / f"{date.year:04d}"
        / f"{date.month:02d}"
        / f"{date}.png"
    )
    fig, _ = plotting.sky_map(
        grid.lon_edges,
        grid.lat_edges,
        exposure,
        "Exposure Map in Galactic Coordinates (Mollweide Projection)",
        outputs.EXPOSURE_LABEL,
    )
    plotting.save(fig, figure)

    result = ensure_dir(outputs.all_sky_dir(paths, energy) / "daily") / f"{date}.npz"
    np.savez(result, lon_edges=grid.lon_edges, lat_edges=grid.lat_edges, exposure_map=exposure)
    log.info("Saved %s and %s", result, figure)


def process_target_day(
    paths: Paths,
    date: datetime.date,
    area: EffectiveAreaTable,
    energy: float,
    target: str,
    target_lon: float,
    target_lat: float,
    side_deg: float,
) -> None:
    """Exposure map of a target region for one day."""
    df = mask_saa(load_day(paths.hk_dir, date))[0]
    if len(df) == 0:
        log.warning("No data found for %s", date)
        return
    res = compute_exposure_map_roi(df, area, energy, target_lon, target_lat, side_deg)
    figure = (
        outputs.target_figure_dir(paths, target, energy)
        / f"{date.year:04d}"
        / f"{date.month:02d}"
        / f"{date}.png"
    )
    outputs.plot_roi(
        res["lon_centers"],
        res["lat_centers"],
        res["exposure"],
        target_lon,
        target_lat,
        target,
        energy,
        figure,
    )
    result = ensure_dir(outputs.target_dir(paths, target, energy) / "daily") / f"{date}.npz"
    np.savez(
        result, lon_edges=res["lon_edges"], lat_edges=res["lat_edges"], exposure_map=res["exposure"]
    )
    log.info("Processed %s: saved to %s, %s", date, figure, result)


def target_dates(
    args: argparse.Namespace, paths: Paths
) -> tuple[list[datetime.date], str, str] | None:
    """Dates to process, summed map suffix and mode ('year' or 'range')."""
    if args.year is not None:
        if args.year.lower() == "all":
            years = range(FIRST_YEAR, LAST_YEAR + 1)
        elif args.year == "2025":
            log.info("Year 2025 excluded from processing.")
            return None
        else:
            years = [args.year]
        return available_dates(paths.hk_dir, years), "", "year"

    if args.start is None or args.end is None:
        raise SystemExit("--target needs either --year or both --start and --end")
    if "all" in (args.start.lower(), args.end.lower()):
        start, end = datetime.date(FIRST_YEAR, 1, 1), datetime.date(LAST_YEAR, 12, 31)
    else:
        start, end = parse_date(args.start), parse_date(args.end)
        if start > end:
            raise SystemExit("Start date must be before end date.")
    dates = [d for d in available_dates(paths.hk_dir) if start <= d <= end]
    return dates, f"_{start}_to_{end}", "range"


def run_target(args: argparse.Namespace, paths: Paths, area: EffectiveAreaTable, energy: float):
    """Daily exposure maps of a target region over a year or a date range, then their sum."""
    selection = target_dates(args, paths)
    if selection is None:
        return
    dates, suffix, mode = selection
    if not dates:
        log.info("No data files found for the requested period.")
        return
    side_deg = args.roi_size if args.roi_size is not None else DEFAULT_ROI_DEG[mode]
    target_lon, target_lat = target_galactic_coordinates(args.target, outputs.par_dir(paths))
    log.info(
        "Processing %d days from %s to %s for %s (l=%.4f, b=%.4f) at %.3f GeV, ROI %g deg",
        len(dates), dates[0], dates[-1], args.target, target_lon, target_lat, energy, side_deg,
    )  # fmt: skip
    for i, date in enumerate(dates):
        if i % 100 == 0:
            log.info("Processing day %d/%d: %s", i + 1, len(dates), date)
        try:
            process_target_day(
                paths, date, area, energy, args.target, target_lon, target_lat, side_deg
            )
        except Exception:
            log.exception("Error processing %s", date)
    sum_target(paths, args.target, energy, target_lon, target_lat, suffix)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--energy-bin", type=int, required=True, help="index of the GIEM energy bin"
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--date", type=parse_date, help="all sky map of one day, YYYY-MM-DD")
    where.add_argument("--target", help="alias (Vela, Crab, Geminga, KM3-230213A) or PSR name")
    parser.add_argument("--year", help="target mode: one year or ALL (2016 to 2024)")
    parser.add_argument("--start", help="target mode: first day YYYY-MM-DD, or ALL")
    parser.add_argument("--end", help="target mode: last day YYYY-MM-DD, or ALL")
    parser.add_argument(
        "--roi-size", type=float, help="ROI side in degrees (default 4 with --year, 6 with --start)"
    )


def run(args: argparse.Namespace, paths: Paths) -> None:
    plotting.use_style()
    energy = outputs.energy_bin_centers(paths)[args.energy_bin]
    area = EffectiveAreaTable.from_npz(paths.effective_area_file)
    if args.date is not None:
        run_all_sky(paths, args.date, area, energy)
    else:
        run_target(args, paths, area, energy)
