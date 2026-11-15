import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GCRS, ITRS, EarthLocation
from astropy.time import Time

from dampe_gamma.exposure import saa
from dampe_gamma.exposure.housekeeping import MET_EPOCH


def test_points_inside_and_outside():
    lon = np.array([-50.0, -30.0, -60.0, 0.0, 100.0, -50.0])
    lat = np.array([-25.0, -30.0, -40.0, 50.0, 0.0, 30.0])
    inside = saa.inside_saa(lon, lat)
    assert inside.tolist() == [True, True, True, False, False, False]


def test_polygon_is_valid():
    assert saa.SAA_POLYGON.is_valid
    assert len(saa.SAA_LON_2016) == len(saa.SAA_LAT_2016) == 30


def hk_at(lon_deg, lat_deg, met_seconds):
    """Housekeeping rows of a satellite at 500 km above given geographic positions."""
    obstime = Time(MET_EPOCH, scale="utc") + met_seconds * u.s
    loc = EarthLocation.from_geodetic(lon_deg * u.deg, lat_deg * u.deg, 500 * u.km)
    itrs = ITRS(loc.get_itrs(obstime).cartesian, obstime=obstime)
    xyz = itrs.transform_to(GCRS(obstime=obstime)).cartesian.xyz.to(u.m).value
    return pd.DataFrame(
        {"second": met_seconds, "microsecond": np.zeros_like(met_seconds), "x": xyz[0],
         "y": xyz[1], "z": xyz[2]}
    )  # fmt: skip


def test_mask_saa_from_gcrs_positions():
    met = np.array([1.6e8, 1.6e8 + 60, 1.6e8 + 120])
    df = hk_at(np.array([-50.0, 10.0, 120.0]), np.array([-25.0, 45.0, -10.0]), met)
    lon, lat = saa.geographic_position(df)
    assert np.allclose(lon, [-50, 10, 120], atol=1e-3)
    assert np.allclose(lat, [-25, 45, -10], atol=0.3)
    kept, valid = saa.mask_saa(df)
    assert valid.tolist() == [False, True, True]
    assert kept["second"].tolist() == met[1:].tolist()
