import datetime

import numpy as np
import pandas as pd
from astropy.time import Time

from dampe_gamma.exposure import housekeeping as hk


def test_met_epoch():
    t = hk.met_to_utc(0, 0)
    assert t.scale == "utc"
    assert t.isot == "2013-01-01T00:00:00.000"


def test_met_seconds_and_microseconds():
    t = hk.met_to_utc(np.array([86400, 3600]), np.array([500000, 0]))
    assert t[0].isot == "2013-01-02T00:00:00.500"
    assert t[1].isot == "2013-01-01T01:00:00.000"


def test_file_layout(tmp_path):
    date = datetime.date(2018, 3, 7)
    path = hk.day_file(tmp_path, date)
    assert path == tmp_path / "2018" / "03" / "07_data_photon.root"
    assert hk.date_from_path(path) == date
    path.parent.mkdir(parents=True)
    path.touch()
    (path.parent / "08_data_photon.root").touch()
    (tmp_path / "2019" / "01").mkdir(parents=True)
    (tmp_path / "2019" / "01" / "01_data_photon.root").touch()
    assert hk.month_files(tmp_path, 2018, 3) == sorted(str(p) for p in path.parent.iterdir())
    assert hk.available_dates(tmp_path, [2018]) == [date, datetime.date(2018, 3, 8)]
    assert len(hk.available_dates(tmp_path)) == 3


def test_restrict_to_day_dedups_and_cuts_at_midnight():
    start = int(round((Time("2018-03-07T00:00:00", scale="utc") - hk.met_to_utc(0, 0)).sec))
    seconds = np.array([start - 1, start, start, start + 5, start + 86399, start + 86401])
    df = pd.DataFrame({"second": seconds, "microsecond": [0, 7, 3, 0, 999999, 0]})
    day = hk.restrict_to_day(df.sample(frac=1, random_state=0), datetime.date(2018, 3, 7))
    assert day["second"].tolist() == [start, start + 5, start + 86399]
    assert day["microsecond"].iloc[0] == 3


def test_day_bounds_span_one_day():
    start, end = hk.day_bounds(datetime.date(2018, 3, 7))
    assert (end - start).sec == 86400
