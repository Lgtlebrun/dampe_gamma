"""Pulsar ephemeris (.par) files and target positions."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

TARGET_ALIASES: dict[str, str] = {
    "Vela": "J0835-4510",
    "Crab": "J0534+2200",
    "Geminga": "J0633+1746",
    "KM3-230213A": "KM3-230213A",
}


def parse_par_file(path: str | Path) -> tuple[str | None, str | None]:
    """RAJ and DECJ strings of a .par file, None when absent."""
    raj = decj = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].upper()
            if key == "RAJ":
                raj = parts[1]
            elif key == "DECJ":
                decj = parts[1]
    return raj, decj


def ra_to_hours(ra: str) -> float:
    """Right ascension HH:MM:SS.s to decimal hours."""
    parts = ra.split(":")
    hours = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0
    seconds = float(parts[2]) if len(parts) > 2 else 0
    return hours + minutes / 60.0 + seconds / 3600.0


def dec_to_degrees(dec: str) -> float:
    """Declination +DD:MM:SS.s to decimal degrees."""
    sign = 1
    if dec.startswith("+"):
        dec = dec[1:]
    elif dec.startswith("-"):
        sign = -1
        dec = dec[1:]
    parts = dec.split(":")
    degrees = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0
    seconds = float(parts[2]) if len(parts) > 2 else 0
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def find_par_file(target: str, par_dir: str | Path) -> Path:
    """Ephemeris file of a target given by alias, pulsar name or path."""
    if Path(target).suffix == ".par" and Path(target).is_file():
        return Path(target)
    name = TARGET_ALIASES.get(target, target)
    par_dir = Path(par_dir)
    candidates = [par_dir / f"{name}.par", *sorted(par_dir.glob(f"{name}_*.par"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .par file for target {target!r} ({name}) in {par_dir}")


def icrs_to_galactic(ra_hours: float, dec_deg: float) -> tuple[float, float]:
    """Galactic (l, b) in degrees, with l wrapped to [-180, 180)."""
    c = SkyCoord(ra=ra_hours * u.hourangle, dec=dec_deg * u.degree, frame="icrs")
    gal = c.transform_to("galactic")
    return float(gal.l.wrap_at(180 * u.deg).degree), float(gal.b.degree)


def target_galactic_coordinates(target: str, par_dir: str | Path) -> tuple[float, float]:
    """Galactic (l, b) in degrees of a target, with l wrapped to [-180, 180)."""
    path = find_par_file(target, par_dir)
    raj, decj = parse_par_file(path)
    if raj is None or decj is None:
        raise ValueError(f"Could not find RAJ and DECJ in {path}")
    return icrs_to_galactic(ra_to_hours(raj), dec_to_degrees(decj))
