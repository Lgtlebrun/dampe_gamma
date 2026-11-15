"""Plotting helpers shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 400


def use_style() -> None:
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI


def save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def energy_cos_theta_map(
    values: np.ndarray,
    energy_edges: np.ndarray,
    cos_theta_edges: np.ndarray,
    title: str,
    label: str,
    **pcolormesh_kwargs,
):
    """Colour map of a quantity tabulated in (energy, cos(theta))."""
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        energy_edges, cos_theta_edges, np.asarray(values).T, shading="auto", **pcolormesh_kwargs
    )
    ax.set_xscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("cos(θ)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=label)
    fig.tight_layout()
    return fig, ax


MOLLWEIDE_LON_LABELS = np.array([150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150])


def sky_map(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
    log_scale: bool = False,
    facecolor: str | None = None,
    lon_label_color: str | None = None,
):
    """Mollweide map of a (n_lon, n_lat) array on 1D galactic lon and lat in radians."""
    from matplotlib.colors import LogNorm

    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="ij")
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    norm = LogNorm() if log_scale else None
    mesh = ax.pcolormesh(
        lon_grid.T, lat_grid.T, np.asarray(values).T, cmap="viridis", shading="auto", norm=norm
    )
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xticklabels(MOLLWEIDE_LON_LABELS)
    if lon_label_color:
        ax.xaxis.set_tick_params(labelcolor=lon_label_color)
    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")
    if facecolor:
        ax.set_facecolor(facecolor)
    ax.grid(True)
    return fig, ax


def roi_map(
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    values: np.ndarray,
    target_lon_deg: float,
    target_lat_deg: float,
    target_name: str,
    title: str,
    label: str,
):
    """Rectangular map of a region of interest, longitude increasing to the left."""
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    ):
        lon_grid, lat_grid = np.meshgrid(
            np.rad2deg(lon_centers), np.rad2deg(lat_centers), indexing="ij"
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        mesh = ax.pcolormesh(lon_grid, lat_grid, values, cmap="viridis", shading="auto")
        ax.plot(target_lon_deg, target_lat_deg, "rx", markersize=10, label=target_name)
        ax.xaxis.set_inverted(True)
        ax.legend(loc="upper right")
        ax.set_xlabel("Galactic Longitude [deg]")
        ax.set_ylabel("Galactic Latitude [deg]")
        ax.set_title(title)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(label, fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        ax.set_aspect("auto")
        ax.grid(True)
        fig.tight_layout()
    return fig, ax
