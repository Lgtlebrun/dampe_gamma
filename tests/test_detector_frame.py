import numpy as np
import pytest

from dampe_gamma.exposure import detector_frame as dfm


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(200, 3)) * 7e6
    vel = rng.normal(size=(200, 3)) * 7e3
    return dfm.satellite_frame(pos, vel), pos


@pytest.mark.parametrize("robust", [True, False])
def test_frame_is_orthonormal_and_right_handed(robust):
    rng = np.random.default_rng(2)
    f = dfm.satellite_frame(rng.normal(size=(50, 3)), rng.normal(size=(50, 3)), robust=robust)
    axes = np.stack([f.x, f.y, f.z], axis=1)
    assert np.allclose(axes @ axes.transpose(0, 2, 1), np.eye(3), atol=1e-12)
    assert np.allclose(np.cross(f.z, f.x), f.y)


def test_z_points_to_nadir(frame):
    f, pos = frame
    assert np.allclose(f.z, -pos / np.linalg.norm(pos, axis=1, keepdims=True))


def test_zenith_target_has_cos_theta_one(frame):
    f, pos = frame
    zenith = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    cos_theta, _ = dfm.track_angles(f, zenith, zenith=True)
    assert np.allclose(cos_theta, 1)
    cos_theta_nadir, _ = dfm.track_angles(f, zenith, zenith=False)
    assert np.allclose(cos_theta_nadir, -1)


def test_grid_angles_match_track_angles(frame):
    f, _ = frame
    directions = np.random.default_rng(3).normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cos_grid = dfm.grid_cos_theta(f, directions, zenith=True)
    phi_grid = dfm.grid_phi_deg(f, directions)
    assert cos_grid.shape == phi_grid.shape == (len(f), 30)
    assert cos_grid.dtype == phi_grid.dtype == np.float32
    for j in (0, 17):
        cos_t, phi_t = dfm.track_angles(f, np.tile(directions[j], (len(f), 1)), zenith=True)
        assert np.allclose(cos_grid[:, j], cos_t, atol=1e-6)
        assert np.allclose(phi_grid[:, j], phi_t, atol=1e-3)
    assert phi_grid.min() >= 0 and phi_grid.max() < 360


def test_velocity_direction_has_phi_zero():
    pos = np.array([[7e6, 0, 0]])
    vel = np.array([[100.0, 7e3, 0]])
    f = dfm.satellite_frame(pos, vel)
    cos_theta, phi = dfm.track_angles(f, np.array([[0.0, 1.0, 0.0]]), zenith=True)
    assert cos_theta == pytest.approx(0, abs=1e-12)
    assert phi[0] == pytest.approx(0, abs=1e-9) or phi[0] == pytest.approx(360)


def test_degenerate_velocity_fallback():
    pos = np.array([[0.0, 7e6, 0.0], [0.0, 3e6, 6e6]])
    vel = pos * 1e-3
    f = dfm.satellite_frame(pos, vel)
    assert np.all(np.isfinite(f.x))
    assert np.allclose(np.linalg.norm(f.x, axis=1), 1)
    assert np.allclose(np.sum(f.x * f.z, axis=1), 0)
