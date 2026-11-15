"""King profile models of the PSF in the scaled angular deviation x_p.

The fitted quantity is the radial density 2 pi x_p P(x_p), so every model below
except `king` integrates to its amplitude over x_p in [0, inf).
"""

from __future__ import annotations

import numpy as np


def king(xp: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """King profile P(x_p), normalised over the plane: integral of 2 pi x P(x) dx is 1."""
    factor1 = 1 / (2 * np.pi * sigma * sigma)
    factor2 = 1 - 1 / gamma
    factor3 = (1 + 0.5 * (1 / gamma) * ((xp / sigma) ** 2)) ** (-gamma)
    return factor1 * factor2 * factor3


def x_king(xp: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """x_p times the King profile."""
    return xp * king(xp, sigma, gamma)


def radial_king(xp: np.ndarray, amplitude: float, sigma: float, gamma: float) -> np.ndarray:
    """Single King term in radial form, 2 pi A x_p P(x_p)."""
    return 2 * np.pi * amplitude * x_king(xp, sigma, gamma)


core_model = radial_king
"""Core component of the two stage fit."""

tail_model = radial_king
"""Tail component of the two stage fit."""


def double_king(
    xp: np.ndarray,
    A: float,
    F: float,
    sigma_c: float,
    gamma_c: float,
    sigma_t: float,
    gamma_t: float,
) -> np.ndarray:
    """Radial double King density with total amplitude A and core fraction F."""
    return (
        2 * np.pi * A * (F * x_king(xp, sigma_c, gamma_c) + (1 - F) * x_king(xp, sigma_t, gamma_t))
    )


def double_king_from_amplitudes(
    xp: np.ndarray,
    A_c: float,
    sigma_c: float,
    gamma_c: float,
    A_t: float,
    sigma_t: float,
    gamma_t: float,
) -> np.ndarray:
    """Double King parametrised by separate core and tail amplitudes."""
    A = A_c + A_t
    F = A_c / A
    return double_king(xp, A, F, sigma_c, gamma_c, sigma_t, gamma_t)
