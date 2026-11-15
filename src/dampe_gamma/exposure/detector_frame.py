"""Satellite reference frame and detector angles of sky directions.

The frame has z along the nadir, x along the velocity made orthogonal to z, and
y = z cross x. The polar angle is measured from the zenith (minus z) for the exposure
maps and from the nadir (plus z) for the observing time, as in the original analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEGENERATE_VELOCITY = 1e-5


@dataclass(frozen=True)
class SatelliteFrame:
    """Unit axes of the satellite frame, one row per sample, in the position frame."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, index) -> SatelliteFrame:
        return SatelliteFrame(self.x[index], self.y[index], self.z[index])


def satellite_frame(
    position: np.ndarray, velocity: np.ndarray, robust: bool = True
) -> SatelliteFrame:
    """Satellite frame from (N, 3) positions and velocities.

    robust (exposure maps) replaces a degenerate velocity by (1, 0, 0) cross z and
    orthogonalises x again. The observing time uses robust=False.
    """
    suz = -position / np.linalg.norm(position, axis=1, keepdims=True)
    sux = velocity - np.sum(velocity * suz, axis=1, keepdims=True) * suz
    if robust:
        too_small = np.linalg.norm(sux, axis=1) < DEGENERATE_VELOCITY
        sux[too_small] = np.cross(np.array([[1, 0, 0]]), suz[too_small])
    sux /= np.linalg.norm(sux, axis=1, keepdims=True)
    suy = np.cross(suz, sux)
    if robust:
        sux = np.cross(suy, suz)
    return SatelliteFrame(sux, suy, suz)


def frame_from_hk(df: pd.DataFrame, robust: bool = True) -> SatelliteFrame:
    position = np.vstack([df["x"].values, df["y"].values, df["z"].values]).T
    velocity = np.vstack([df["vx"].values, df["vy"].values, df["vz"].values]).T
    return satellite_frame(position, velocity, robust=robust)


def _axis(frame: SatelliteFrame, zenith: bool) -> np.ndarray:
    return -frame.z if zenith else frame.z


def grid_cos_theta(frame: SatelliteFrame, directions: np.ndarray, *, zenith: bool) -> np.ndarray:
    """cos(theta) in float32 of M unit directions at N samples, shape (N, M)."""
    return (_axis(frame, zenith) @ directions.T).astype(np.float32)


def grid_phi_deg(frame: SatelliteFrame, directions: np.ndarray) -> np.ndarray:
    """Azimuth in [0, 360) degrees, float32, of M unit directions at N samples."""
    t_dot_x = (frame.x @ directions.T).astype(np.float32)
    t_dot_y = (frame.y @ directions.T).astype(np.float32)
    return np.rad2deg(np.mod(np.arctan2(t_dot_y, t_dot_x), 2 * np.pi))


def track_angles(
    frame: SatelliteFrame, directions: np.ndarray, *, zenith: bool
) -> tuple[np.ndarray, np.ndarray]:
    """cos(theta) and azimuth in degrees of one direction per sample, shape (N,)."""
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    cos_theta = np.sum(unit * _axis(frame, zenith), axis=1)
    phi = np.mod(
        np.arctan2(np.sum(unit * frame.y, axis=1), np.sum(unit * frame.x, axis=1)), 2 * np.pi
    )
    return cos_theta, np.degrees(phi)
