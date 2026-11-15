"""Effective area from the selection efficiency of the weighted MC sample.

The effective area is the fraction of simulated photons surviving the selection, times
the geometric area of the generation surface, tabulated in (E, cos(theta), phi).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dampe_gamma import plotting
from dampe_gamma.binning import IrfBinning, cos_theta_edges, energy_edges, phi_edges_deg
from dampe_gamma.config import Paths
from dampe_gamma.io import browse_files, ensure_dir, read_root_columns, summarize

log = logging.getLogger(__name__)

GEOMETRIC_AREA_CM2 = 6e4
TREE = "MC"
COLUMNS = (
    "tt_ekin",
    "weight",
    "tt_theta",
    "tt_phi",
    "theta_BestSTK",
    "tt_bgoTotalEcorr_GeV",
    "tt_isHET",
)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add true energy in GeV, cos(theta) and phi in degrees to raw MC columns."""
    df = df.loc[(df["tt_theta"] < np.pi) & (df["tt_theta"] > -np.pi)].copy()
    df["E_true"] = df["tt_ekin"] * 1e-3
    df["cosTheta"] = np.cos(df["tt_theta"])
    df["phi_True_deg"] = np.rad2deg(np.mod(df["tt_phi"], 2 * np.pi))
    return df


def _ratio(accepted: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.divide(accepted, total, out=np.zeros_like(accepted, dtype=float), where=total != 0)


@dataclass(frozen=True)
class Efficiency:
    """Selection efficiency, unweighted and weighted, at three levels of projection."""

    binning: IrfBinning
    full: np.ndarray
    full_weighted: np.ndarray
    energy_cos_theta: np.ndarray
    energy_cos_theta_weighted: np.ndarray
    energy: np.ndarray
    energy_weighted: np.ndarray
    energy_error: np.ndarray


def compute_efficiency(
    total: pd.DataFrame, selected: pd.DataFrame, binning: IrfBinning
) -> Efficiency:
    """Ratio of selected to generated events in each bin, with binomial errors in energy."""
    axes = ("E_true", "cosTheta", "phi_True_deg")
    edges = [binning.energy, binning.cos_theta, binning.phi_deg]

    def histograms(df: pd.DataFrame, weighted: bool):
        w = df["weight"] if weighted else None
        full, _ = np.histogramdd(df[list(axes)].to_numpy(), bins=edges, weights=w)
        two_d, _, _ = np.histogram2d(df["E_true"], df["cosTheta"], bins=edges[:2], weights=w)
        one_d, _ = np.histogram(df["E_true"], bins=edges[0], weights=w)
        return full, two_d, one_d

    tot, tot_w = histograms(total, False), histograms(total, True)
    sel, sel_w = histograms(selected, False), histograms(selected, True)
    ratios = [_ratio(s, t) for s, t in zip(sel, tot, strict=True)]
    ratios_w = [_ratio(s, t) for s, t in zip(sel_w, tot_w, strict=True)]

    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.nan_to_num(np.sqrt(ratios[2] * (1 - ratios[2]) / tot[2]))

    return Efficiency(
        binning, ratios[0], ratios_w[0], ratios[1], ratios_w[1], ratios[2], ratios_w[2], error
    )


def save_effective_area(path: Path, effective_area: np.ndarray, binning: IrfBinning) -> None:
    np.savez(
        path,
        E_bins_edge=binning.energy,
        theta_bins_edge=binning.cos_theta,
        phi_bins_edge=binning.phi_deg,
        effArea=effective_area,
    )


def load_effective_area(path: str | Path) -> tuple[np.ndarray, IrfBinning]:
    """Read a table written by save_effective_area."""
    with np.load(path) as data:
        binning = IrfBinning(data["E_bins_edge"], data["theta_bins_edge"], data["phi_bins_edge"])
        return data["effArea"], binning


def plot_efficiency(
    energy_centers: np.ndarray, efficiency: np.ndarray, error: np.ndarray | None = None
):
    fig, ax = plt.subplots(figsize=(8, 5))
    if error is not None:
        ax.errorbar(
            energy_centers,
            efficiency * 100,
            yerr=error * 100,
            fmt="o-",
            color="green",
            ecolor="green",
            capsize=3,
            label="Binomial errors",
        )
    else:
        ax.plot(energy_centers, efficiency * 100, "o-", color="green", label="Efficiency")
    ax.set_xscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Efficiency (accepted / total) [%]")
    ax.set_title("Detection Efficiency vs Energy")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e", "--exponent", type=float, default=0.0, help="spectral index of the MC weights"
    )
    parser.add_argument("--energy-binning", choices=("log", "giem"), default="log")
    parser.add_argument("--n-energy-bins", type=int, default=40, help="used by the log binning")
    parser.add_argument("--n-cos-theta-bins", type=int, default=10)
    parser.add_argument("--n-phi-bins", type=int, default=360)


def run(args: argparse.Namespace, paths: Paths) -> None:
    plotting.use_style()
    binning = IrfBinning(
        energy_edges(args.energy_binning, args.n_energy_bins),
        cos_theta_edges(args.n_cos_theta_bins),
        phi_edges_deg(args.n_phi_bins),
    )
    tables = []
    for directory in (paths.scaled_mc_dir(args.exponent), paths.filtered_mc_dir(args.exponent)):
        files = browse_files(directory / "*.root")
        if not files:
            raise FileNotFoundError(f"No ROOT files in {directory}")
        df = preprocess(read_root_columns(files, TREE, COLUMNS))
        log.info(summarize(df, directory.name))
        tables.append(df)

    eff = compute_efficiency(*tables, binning)
    area = eff.full * GEOMETRIC_AREA_CM2
    area_w = eff.full_weighted * GEOMETRIC_AREA_CM2
    per_sr_2d = eff.energy_cos_theta * GEOMETRIC_AREA_CM2 / (2 * np.pi)
    per_sr_2d_w = eff.energy_cos_theta_weighted * GEOMETRIC_AREA_CM2 / (2 * np.pi)

    figure_dir = ensure_dir(paths.figures_dir / "effArea" / "script" / f"exp{args.exponent:.1f}")
    result_dir = ensure_dir(paths.results_dir / "effArea")
    label = "Effective Area [cm²]"
    maps = {
        "effectiveArea_unweighted": (per_sr_2d, "Effective Area"),
        "effectiveArea_weighted": (per_sr_2d_w, "Weighted Effective Area"),
        "effectiveArea_weighted_averaged_phi": (area_w.mean(axis=2), "Weighted Effective Area"),
    }
    for name, (values, title) in maps.items():
        fig, _ = plotting.energy_cos_theta_map(
            values, binning.energy, binning.cos_theta, title, label, cmap="viridis"
        )
        plotting.save(fig, figure_dir / f"{name}.png")

    centers = binning.energy_centers
    plotting.save(
        plot_efficiency(centers, eff.energy, eff.energy_error), figure_dir / "efficiency.png"
    )
    plotting.save(
        plot_efficiency(centers, eff.energy_weighted), figure_dir / "efficiency_weighted.png"
    )

    save_effective_area(result_dir / "effectiveArea_unweighted.npz", area, binning)
    save_effective_area(result_dir / "effectiveArea_weighted.npz", area_w, binning)
    log.info("Effective areas written to %s", result_dir)
