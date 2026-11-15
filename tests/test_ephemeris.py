import pytest

from dampe_gamma import ephemeris
from dampe_gamma.config import REPO_ROOT

PAR_DIR = REPO_ROOT / "data" / "par"


def test_parse_vela_par_file():
    ra, dec = ephemeris.parse_par_file(PAR_DIR / "J0835-4510_tnfit.par")
    assert ra == "8:35:20.61149000"
    assert dec == "-45:10:34.87510000"


def test_sexagesimal_conversion():
    assert ephemeris.ra_to_hours("06:30:00") == pytest.approx(6.5)
    assert ephemeris.dec_to_degrees("-07:48:00.00") == pytest.approx(-7.8)
    assert ephemeris.dec_to_degrees("+17:46:12") == pytest.approx(17.77)


def test_vela_galactic_coordinates():
    lon, lat = ephemeris.target_galactic_coordinates("Vela", PAR_DIR)
    assert -180 <= lon < 180
    assert lon % 360 == pytest.approx(263.55, abs=0.01)
    assert lat == pytest.approx(-2.79, abs=0.01)


@pytest.mark.parametrize(
    "target, l_deg, b_deg",
    [("Crab", 184.56, -5.78), ("Geminga", 195.13, 4.27), ("KM3-230213A", 216.06, -11.13)],
)
def test_aliases(target, l_deg, b_deg):
    lon, lat = ephemeris.target_galactic_coordinates(target, PAR_DIR)
    assert lon % 360 == pytest.approx(l_deg, abs=0.01)
    assert lat == pytest.approx(b_deg, abs=0.01)


def test_pulsar_name_and_path_are_accepted():
    by_name = ephemeris.find_par_file("J1709-4429", PAR_DIR)
    assert by_name.name == "J1709-4429_tnfit.par"
    assert ephemeris.find_par_file(str(by_name), PAR_DIR) == by_name


def test_unknown_target_raises():
    with pytest.raises(FileNotFoundError):
        ephemeris.find_par_file("NotAPulsar", PAR_DIR)
