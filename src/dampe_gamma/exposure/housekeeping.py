"""Satellite housekeeping data: file layout, mission time and daily tables."""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from dampe_gamma.io import browse_files, read_root_columns

log = logging.getLogger(__name__)

HK_TREE = "HousekeepingData/SatStatus"
HK_BRANCHES = {
    "DmpHKDSatStatus.fSecond": "second",
    "DmpHKDSatStatus.fMicrosecond": "microsecond",
    "DmpHKDSatStatus.fPosition.fX": "x",
    "DmpHKDSatStatus.fPosition.fY": "y",
    "DmpHKDSatStatus.fPosition.fZ": "z",
    "DmpHKDSatStatus.fVelocity.fX": "vx",
    "DmpHKDSatStatus.fVelocity.fY": "vy",
    "DmpHKDSatStatus.fVelocity.fZ": "vz",
}
MET_EPOCH = "2013-01-01T00:00:00"
FILE_SUFFIX = "_data_photon.root"
_DATE_IN_PATH = re.compile(r"/(\d{4})/(\d{2})/(\d{2})_")


def met_to_utc(seconds, microseconds) -> Time:
    """Mission elapsed time (seconds since 2013-01-01 UTC) to astropy Time."""
    delta_sec = TimeDelta(np.asarray(seconds), format="sec")
    delta_us = TimeDelta(np.asarray(microseconds) * 1e-6, format="sec")
    return Time(MET_EPOCH, scale="utc") + delta_sec + delta_us


def hk_times(df: pd.DataFrame) -> Time:
    """UTC time of each housekeeping sample."""
    return met_to_utc(df["second"].to_numpy(), df["microsecond"].to_numpy())


def day_bounds(date: datetime.date | str) -> tuple[Time, Time]:
    """Start and end of a UTC day, the end being start plus one Julian day."""
    start = Time(f"{date}T00:00:00", scale="utc")
    return start, start + TimeDelta(1, format="jd")


def parse_date(text: str) -> datetime.date:
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()


def day_file(hk_dir: str | Path, date: datetime.date) -> Path:
    """Housekeeping file of one day: hk_dir/YYYY/MM/DD_data_photon.root."""
    return Path(hk_dir) / f"{date.year:04d}" / f"{date.month:02d}" / f"{date.day:02d}{FILE_SUFFIX}"


def month_files(hk_dir: str | Path, year: int, month: int) -> list[str]:
    return browse_files(Path(hk_dir) / f"{year}" / f"{month:02d}" / f"*{FILE_SUFFIX}")


def date_from_path(path: str | Path) -> datetime.date | None:
    match = _DATE_IN_PATH.search(str(path))
    if match is None:
        return None
    return datetime.date(*(int(g) for g in match.groups()))


def available_dates(
    hk_dir: str | Path, years: Iterable[int | str] | None = None
) -> list[datetime.date]:
    """Sorted dates having a housekeeping file, optionally restricted to some years."""
    year_globs = ["[0-9][0-9][0-9][0-9]"] if years is None else [str(y) for y in years]
    files = browse_files(Path(hk_dir) / y / "[0-1][0-9]" / f"*{FILE_SUFFIX}" for y in year_globs)
    dates = set()
    for f in files:
        date = date_from_path(f)
        if date is None:
            log.warning("Could not extract a date from %s", f)
            continue
        dates.add(date)
    return sorted(dates)


def read_hk(files: Sequence[str]) -> pd.DataFrame:
    """Position and velocity samples of housekeeping files, renamed to short columns."""
    df = read_root_columns(list(files), HK_TREE, list(HK_BRANCHES))
    return df.rename(columns=HK_BRANCHES)


def restrict_to_day(df: pd.DataFrame, date: datetime.date | str) -> pd.DataFrame:
    """Time ordered samples of one UTC day, keeping the first sample of each second."""
    start, end = day_bounds(date)
    df = df.sort_values(by=["second", "microsecond"]).drop_duplicates("second", keep="first")
    utc = hk_times(df)
    in_day = (utc >= start) & (utc < end)
    return df.loc[in_day].reset_index(drop=True)


def load_day(
    hk_dir: str | Path, date: datetime.date, include_next_day: bool = True
) -> pd.DataFrame:
    """Samples of one UTC day, read from its file and, by default, the next day file."""
    patterns = [day_file(hk_dir, date)]
    if include_next_day:
        patterns.append(day_file(hk_dir, date + datetime.timedelta(days=1)))
    files = sorted(sum((browse_files(p) for p in patterns), []))
    if not files:
        raise FileNotFoundError(f"No housekeeping file for {date} in {hk_dir}")
    return restrict_to_day(read_hk(files), date)
