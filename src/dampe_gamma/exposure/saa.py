"""South Atlantic Anomaly mask of the satellite orbit."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from shapely.geometry import Polygon

from dampe_gamma.exposure.housekeeping import hk_times

try:
    from shapely import contains_xy as _contains_xy
except ImportError:
    from shapely.vectorized import contains as _contains_xy

SAA_LAT_2016 = (
    -8.136145, -5.522644, -2.909143, -0.295642, 2.317859, 4.931360,
    -1.170294, -5.924736, -10.080192, -13.636662, -16.594145, -18.952644,
    -32.560543, -34.555897, -36.591114, -38.661793, -40.743484, -42.791676,
    -44.741795, -46.509228, -47.989288, -49.057251, -49.568325, -49.357666,
    -48.240376, -46.011497, -42.446022, -37.298885, -30.304964, -21.179083,
)  # fmt: skip
SAA_LON_2016 = (
    -95.0, -84.0, -73.0, -62.0, -51.0, -40.0,
    -35.0, -29.0, -23.0, -17.0, -11.0, -5.0,
    -5.0, -10.0, -15.0, -20.0, -25.0, -30.0,
    -35.0, -40.0, -45.0, -50.0, -55.0, -60.0,
    -65.0, -70.0, -75.0, -80.0, -85.0, -90.0,
)  # fmt: skip
SAA_POLYGON = Polygon(list(zip(SAA_LON_2016, SAA_LAT_2016, strict=True)))
"""SAA contour of 2016 in geographic (longitude, latitude) degrees."""


def geographic_position(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Geodetic longitude and latitude in degrees of GCRS positions given in metres."""
    obstime = hk_times(df)
    positions = CartesianRepresentation(
        x=df["x"].to_numpy() * u.m, y=df["y"].to_numpy() * u.m, z=df["z"].to_numpy() * u.m
    )
    itrs = GCRS(positions, obstime=obstime).transform_to(ITRS(obstime=obstime))
    geodetic = itrs.earth_location.to_geodetic()
    return geodetic.lon.to(u.deg).value, geodetic.lat.to(u.deg).value


def inside_saa(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    return np.asarray(_contains_xy(SAA_POLYGON, lon_deg, lat_deg), dtype=bool)


def mask_saa(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Samples outside the SAA and the boolean mask of kept samples."""
    valid = ~inside_saa(*geographic_position(df))
    return df[valid].reset_index(drop=True), valid
