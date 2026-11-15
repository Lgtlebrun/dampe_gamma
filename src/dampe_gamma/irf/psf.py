"""Point spread function: angular resolution and double King fit per (energy, cos theta) bin.

The angular error of each event is scaled by the 68% containment radius of its bin,
x_p = angle / S_p, and the x_p distribution of every bin is fitted with a double King
profile. The fit runs in two stages (core on small x_p, then tail on the residuals) over
a scan of core percentiles, and falls back to a direct six parameter fit when the best
reduced chi2 is too large.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dampe_gamma import plotting
from dampe_gamma.binning import (
    E_MAX_GEV,
    IrfBinning,
    cos_theta_edges,
    energy_edges,
    geometric_centers,
)
from dampe_gamma.config import Paths
from dampe_gamma.io import browse_files, ensure_dir, read_root_columns, summarize
from dampe_gamma.irf.psf_models import core_model, double_king_from_amplitudes, tail_model
from dampe_gamma.stats import (
    COS_THETA_BIN_COLUMN,
    E_BIN_COLUMN,
    DensityHistogram,
    assign_bins,
    chisq,
    containment_radius,
    density_histogram,
    near_bound_messages,
    table_to_grid,
    weighted_containment_radius,
)

log = logging.getLogger(__name__)

TREE = "MC"
INPUT_COLUMNS = (
    "tt_ekin",
    "tt_theta",
    "tt_phi",
    "weight",
    "tt_isHET",
    "theta_BestSTK",
    "phi_BestSTK",
)
N_ENERGY_BINS = 8
N_COS_THETA_BINS = 5
MIN_THETA_RAD = 1e-7
CONTAINMENT_PERCENT = 68
RESOLUTION_COLUMN = "angular_resolution"

N_PARAMS = 6
CORE_PERCENTILES = np.linspace(20, 80, 13)
FALLBACK_CHI2_RED = 2.5
MIN_XP_CENTER = 1e-2
CORE_P0_SIGMA_GAMMA = (0.5, 2.0)
CORE_BOUNDS = ([0, 0.01, 0.1], [np.inf, 15, 50])
CORE_MAXFEV = 30000
TAIL_P0_SIGMA_GAMMA = (1.0, 2.0)
TAIL_BOUNDS = ([0, 0.001, 0.1], [np.inf, 50, 50])
FULL_MAXFEV = 100000
MIN_XP_BINS, MAX_XP_BINS, ENTRIES_PER_XP_BIN = 10, 30, 5

CORE_NAMES = ("A_core", "sigma_c", "gamma_c")
TAIL_NAMES = ("A_tail", "sigma_t", "gamma_t")
FULL_NAMES = ("A_c", "sigma_c", "gamma_c", "A_t", "sigma_t", "gamma_t")
RECORD_PARAMS = ("A_c", "sigma_c", "gamma_c", "A_t", "sigma_t", "gamma_t")


def default_binning(
    energy_binning: str = "log",
    n_energy_bins: int = N_ENERGY_BINS,
    n_cos_theta_bins: int = N_COS_THETA_BINS,
) -> IrfBinning:
    """PSF binning: log or GIEM energy edges and linear cos(theta) edges in [0.5, 1]."""
    return IrfBinning(
        energy_edges(energy_binning, n_energy_bins), cos_theta_edges(n_cos_theta_bins)
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """True energy in GeV, angles in radians, cos(theta), and removal of degenerate directions."""
    df = df.copy()
    df["E_true"] = df["tt_ekin"] * 1e-3
    df = df.rename(columns={"tt_theta": "theta_True", "tt_phi": "phi_True"})
    df = df[df["E_true"] <= E_MAX_GEV].copy()
    df["cosTheta"] = np.cos(df["theta_True"])
    df["theta_BestSTK"] = np.deg2rad(df["theta_BestSTK"])
    df["phi_BestSTK"] = np.deg2rad(df["phi_BestSTK"])
    return df.loc[(df["theta_True"] > MIN_THETA_RAD) & (df["theta_BestSTK"] > MIN_THETA_RAD)]


def _unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1
    )


def angular_error(
    theta_true: np.ndarray, phi_true: np.ndarray, theta_reco: np.ndarray, phi_reco: np.ndarray
) -> np.ndarray:
    """Angle in radians between true and reconstructed directions."""
    dot = np.sum(_unit_vectors(theta_true, phi_true) * _unit_vectors(theta_reco, phi_reco), axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def compute_angular_error(df: pd.DataFrame) -> pd.DataFrame:
    """Add the angular error in radians and degrees."""
    df = df.copy()
    df["angular_error"] = angular_error(
        df["theta_True"].to_numpy(),
        df["phi_True"].to_numpy(),
        df["theta_BestSTK"].to_numpy(),
        df["phi_BestSTK"].to_numpy(),
    )
    df["angular_error_degrees"] = df["angular_error"] * 180 / np.pi
    return df


def compute_resolution(
    df: pd.DataFrame, binning: IrfBinning
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """68% containment radius per bin, unweighted and weighted; df gets the weighted one."""
    df = assign_bins(df, binning)
    grouped = df.groupby([E_BIN_COLUMN, COS_THETA_BIN_COLUMN])

    resolution = (
        grouped["angular_error_degrees"]
        .apply(containment_radius, CONTAINMENT_PERCENT)
        .reset_index(name=RESOLUTION_COLUMN)
    )
    resolution_w = (
        grouped[["angular_error_degrees", "weight"]]
        .apply(
            lambda g: weighted_containment_radius(
                g["angular_error_degrees"].values, g["weight"].values, CONTAINMENT_PERCENT
            )
        )
        .reset_index(name=RESOLUTION_COLUMN)
    )
    df = df.merge(resolution_w, on=[E_BIN_COLUMN, COS_THETA_BIN_COLUMN], how="left")
    return df, resolution, resolution_w


@dataclass
class KingFit:
    """Result of a double King fit on one x_p histogram."""

    params: dict[str, float]
    chi2: float
    chi2_red: float
    p_value: float
    model_total: np.ndarray
    core_fit: np.ndarray
    tail_fit: np.ndarray
    method: str
    bound_warnings: list[str] = field(default_factory=list)
    histogram: DensityHistogram | None = None


def two_stage_fit(
    xp: np.ndarray, data: np.ndarray, error: np.ndarray, percentile_cutoff: float = 60
) -> KingFit:
    """Fit the core below a percentile of x_p, then the tail on the positive residuals."""
    core = xp < np.percentile(xp, percentile_cutoff)
    p0_core = [np.max(data[core]), *CORE_P0_SIGMA_GAMMA]
    try:
        popt_core, _ = curve_fit(
            core_model,
            xp[core],
            data[core],
            sigma=error[core],
            p0=p0_core,
            bounds=CORE_BOUNDS,
            maxfev=CORE_MAXFEV,
        )
    except Exception as err:
        raise RuntimeError(f"Core fitting failed: {err}") from err
    warnings = near_bound_messages(popt_core, CORE_BOUNDS, CORE_NAMES, "core fit")
    core_fit = core_model(xp, *popt_core)

    residuals = data - core_fit
    residuals[residuals < 0] = 0
    tail = xp > np.percentile(xp, 100 - percentile_cutoff)
    p0_tail = [np.max(residuals[tail]), *TAIL_P0_SIGMA_GAMMA]
    try:
        popt_tail, _ = curve_fit(
            tail_model, xp[tail], residuals[tail], sigma=error[tail], p0=p0_tail, bounds=TAIL_BOUNDS
        )
    except Exception as err:
        raise RuntimeError(f"Tail fitting failed: {err}") from err
    warnings += near_bound_messages(popt_tail, TAIL_BOUNDS, TAIL_NAMES, "tail fit")
    tail_fit = tail_model(xp, *popt_tail)

    A_core, sigma_c, gamma_c = popt_core
    A_tail, sigma_t, gamma_t = popt_tail
    A_total = A_core + A_tail
    model_total = core_fit + tail_fit
    chi2_val, chi2_red, p_val = chisq(data, model_total, error, len(xp) - N_PARAMS)
    params = {
        "A": A_total,
        "F": A_core / A_total,
        "sigma_c": sigma_c,
        "gamma_c": gamma_c,
        "sigma_t": sigma_t,
        "gamma_t": gamma_t,
    }
    return KingFit(
        params, chi2_val, chi2_red, p_val, model_total, core_fit, tail_fit, "two_stage", warnings
    )


def full_fit_p0_bounds(x: np.ndarray, y: np.ndarray) -> tuple[list, tuple[list, list]]:
    """Data driven starting point and bounds of the six parameter fit."""
    y_max, x_max = np.max(y), np.max(x)
    p0 = [0.6 * y_max, x_max / 5, 2.0, 0.4 * y_max, x_max / 2, 2.0]
    lower = [0, 0.01, 0.5, 0, 0.01, 0.5]
    upper = [10 * y_max, x_max, 15, 10 * y_max, 2 * x_max, 15]
    return p0, (lower, upper)


def full_fit(xp: np.ndarray, data: np.ndarray, error: np.ndarray) -> KingFit:
    """Fit all six double King parameters at once."""
    p0, bounds = full_fit_p0_bounds(xp, data)
    try:
        popt, _ = curve_fit(
            double_king_from_amplitudes,
            xp,
            data,
            sigma=error,
            p0=p0,
            bounds=bounds,
            maxfev=FULL_MAXFEV,
        )
    except Exception as err:
        raise RuntimeError(f"Full double King fit failed: {err}") from err
    warnings = near_bound_messages(popt, bounds, FULL_NAMES, "full fit")

    model = double_king_from_amplitudes(xp, *popt)
    chi2_val, chi2_red, p_val = chisq(data, model, error, len(xp) - len(popt))
    A_c, sigma_c, gamma_c, A_t, sigma_t, gamma_t = popt
    A_total = A_c + A_t
    params = {
        "A": A_total,
        "F": A_c / A_total,
        "A_c": A_c,
        "sigma_c": sigma_c,
        "gamma_c": gamma_c,
        "A_t": A_t,
        "sigma_t": sigma_t,
        "gamma_t": gamma_t,
    }
    core_fit = core_model(xp, A_c, sigma_c, gamma_c)
    tail_fit = tail_model(xp, A_t, sigma_t, gamma_t)
    return KingFit(params, chi2_val, chi2_red, p_val, model, core_fit, tail_fit, "full", warnings)


def xp_histogram(xp: np.ndarray, bins: np.ndarray) -> DensityHistogram:
    """Normalised x_p histogram restricted to filled bins above MIN_XP_CENTER."""
    hist = density_histogram(xp, bins)
    return hist.masked((hist.counts > 0) & (hist.centers > MIN_XP_CENTER))


def fit_bin_two_stage(
    xp: np.ndarray, bins: np.ndarray, fallback_threshold: float = FALLBACK_CHI2_RED
) -> KingFit:
    """Two stage fit over the core percentile scan, with a full fit fallback on poor chi2."""
    hist = xp_histogram(xp, bins)
    x, y, err = hist.centers, hist.density, hist.errors

    best, best_chi2_red, last_error = None, np.inf, None
    for percentile in CORE_PERCENTILES:
        try:
            result = two_stage_fit(x, y, err, percentile_cutoff=percentile)
        except Exception as error:
            log.debug("Two stage fit failed at percentile %s: %s", percentile, error)
            last_error = error
            continue
        if np.abs(result.chi2_red - 1) < np.abs(best_chi2_red - 1):
            best, best_chi2_red = result, result.chi2_red

    if best is None or best_chi2_red > fallback_threshold:
        log.info("Two stage fit unsatisfactory, trying the full double King fit")
        try:
            result = full_fit(x, y, err)
            log.info("Full fit chi2_red = %.3f", result.chi2_red)
            if result.chi2_red < best_chi2_red:
                best = result
        except Exception as error:
            log.warning("Full double King fit also failed: %s", error)
            if best is None:
                raise last_error from error
    if best is None:
        raise RuntimeError("No double King fit converged")

    best.histogram = hist
    return best


def xp_bins(xp: np.ndarray) -> np.ndarray:
    """Log spaced x_p edges, between 10 and 30 of them depending on the bin population."""
    n_edges = min(MAX_XP_BINS, max(MIN_XP_BINS, len(xp) // ENTRIES_PER_XP_BIN))
    return np.logspace(np.log10(np.min(xp)), np.log10(np.max(xp)), n_edges)


def _empty_record(e_bin: int, cos_theta_bin: int, n_entries: int) -> dict:
    record = {E_BIN_COLUMN: e_bin, COS_THETA_BIN_COLUMN: cos_theta_bin}
    record.update(dict.fromkeys(RECORD_PARAMS, np.nan))
    record.update(chi2=np.nan, ndof=np.nan, chi2_ndof=np.nan, p_value=np.nan, n_entries=n_entries)
    return record


def _fit_record(e_bin: int, cos_theta_bin: int, fit: KingFit, n_entries: int) -> dict:
    p = fit.params
    return {
        E_BIN_COLUMN: e_bin,
        COS_THETA_BIN_COLUMN: cos_theta_bin,
        "A_c": p["A"] * p["F"],
        "sigma_c": p["sigma_c"],
        "gamma_c": p["gamma_c"],
        "A_t": p["A"] * (1 - p["F"]),
        "sigma_t": p["sigma_t"],
        "gamma_t": p["gamma_t"],
        "chi2": fit.chi2,
        "ndof": len(fit.histogram.centers) - N_PARAMS,
        "chi2_ndof": fit.chi2_red,
        "p_value": fit.p_value,
        "n_entries": n_entries,
    }


def plot_fit_result(
    ax,
    xp: np.ndarray,
    bins: np.ndarray,
    fit: KingFit | None,
    label: str = "",
    show_legend: bool = True,
) -> None:
    """Draw the normalised x_p histogram and, when available, the fitted components."""
    hist = fit.histogram if fit is not None else xp_histogram(xp, bins)
    ax.errorbar(hist.centers, hist.density, yerr=hist.errors, fmt="o", label="Data")
    if fit is None:
        ax.text(0.1, 0.9, "Fit Failed", ha="center", va="center", color="red")
        return
    ax.plot(hist.centers, fit.core_fit, "r--", label="Core Fit")
    ax.plot(hist.centers, fit.tail_fit, "b--", label="Tail Fit")
    ax.plot(hist.centers, fit.model_total, "g-", label="Total Fit")
    if show_legend:
        ax.legend(fontsize=8, loc="best")
    if label:
        ax.text(
            0.05,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7},
        )


def _style_xp_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x_p$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10)


def fit_all_bins(
    df: pd.DataFrame, binning: IrfBinning, results_dir: Path, figures_dir: Path
) -> pd.DataFrame:
    """Fit every (energy, cos theta) bin, save per bin and overview figures, return parameters."""
    n_energy, n_cos_theta = binning.shape
    fig, axes = plt.subplots(
        n_energy,
        n_cos_theta,
        figsize=(n_energy * 4, n_cos_theta * 3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(r"Scaled Angular Deviation $x_p$ in Each (E_bin, θ_bin)", fontsize=16)
    bin_plots_dir = ensure_dir(results_dir / "bin_plots")

    records = []
    for e_bin in range(n_energy):
        for cos_theta_bin in range(n_cos_theta):
            subset = df[(df[E_BIN_COLUMN] == e_bin) & (df[COS_THETA_BIN_COLUMN] == cos_theta_bin)]
            if subset.empty:
                log.info("Empty bin E_bin=%d, cos_theta_bin=%d", e_bin, cos_theta_bin)
                records.append(_empty_record(e_bin, cos_theta_bin, 0))
                continue

            xp = subset["x_p"].to_numpy()
            bins = xp_bins(xp)
            title = f"E_bin={e_bin}, θ_bin={cos_theta_bin}"
            try:
                fit = fit_bin_two_stage(xp, bins)
                records.append(_fit_record(e_bin, cos_theta_bin, fit, len(subset)))
                label = f"$\\chi^2$/ndof = {fit.chi2_red:.2f}"
                log.info("%s: %s fit, chi2/ndof = %.2f", title, fit.method, fit.chi2_red)
            except RuntimeError as err:
                log.warning("Fit failed for %s: %s", title, err)
                fit, label = None, "No fit (failed)"
                records.append(_empty_record(e_bin, cos_theta_bin, len(subset)))

            ax = axes[e_bin, cos_theta_bin]
            _style_xp_axis(ax, title, r"$2\pi\, x_p \,P(x_p)$")
            plot_fit_result(ax, xp, bins, fit, label, show_legend=False)

            fig_single, ax_single = plt.subplots(figsize=(5, 4))
            _style_xp_axis(ax_single, title, r"$2\pi\,\times\,P(x_p)$")
            plot_fit_result(ax_single, xp, bins, fit, label)
            fig_single.tight_layout()
            stem = bin_plots_dir / f"E{e_bin}_theta{cos_theta_bin}"
            fig_single.savefig(f"{stem}.png")
            fig_single.savefig(f"{stem}.eps", format="eps")
            plt.close(fig_single)

    fig.tight_layout()
    fig.savefig(results_dir / "xp_HET_fits_allbins.png")
    ensure_dir(figures_dir)
    fig.savefig(figures_dir / "xp_HET_fits_allbins.eps", format="eps")
    plotting.save(fig, figures_dir / "xp_HET_fits_allbins.png")
    return pd.DataFrame(records)


def plot_angular_error(df: pd.DataFrame, path: Path) -> None:
    """Angular error histogram, and its normalised density on log axes."""
    column = "angular_error_degrees"
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=50, range=(0, 10), histtype="step", color="green", label="Unweighted")
    ax.hist(
        df[column],
        bins=50,
        range=(0, 10),
        weights=df["weight"],
        histtype="step",
        color="blue",
        label="Weighted",
    )
    ax.set_title("Angular Error Distribution")
    ax.set_xlabel("Angular Error (degrees)")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plotting.save(fig, path)

    bins = np.logspace(np.log10(0.01), np.log10(10), 50)
    centers = geometric_centers(bins)
    widths = np.diff(bins)
    fig, ax = plt.subplots(figsize=(8, 6))
    for weights, color, label in (
        (None, "blue", "Unweighted"),
        (df["weight"], "green", "Weighted"),
    ):
        hist, _ = np.histogram(df[column], bins=bins, weights=weights)
        ax.plot(
            centers, hist / np.sum(hist) / widths, drawstyle="steps-mid", color=color, label=label
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Angular Error (degrees)")
    ax.set_ylabel("Probability Density (per degree)")
    ax.set_title("Normalized PSF: Angular Error Distribution")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    fig.tight_layout()
    plotting.save(fig, path.with_name(path.stem + "_density.png"))


def plot_resolution(resolution: pd.DataFrame, binning: IrfBinning, title: str, path: Path) -> None:
    grid = table_to_grid(resolution, binning.shape, RESOLUTION_COLUMN)
    fig, _ = plotting.energy_cos_theta_map(
        grid,
        binning.energy,
        binning.cos_theta,
        title,
        "Angular Resolution (degrees)",
        cmap="viridis",
    )
    plotting.save(fig, path)


def save_binning(path: Path, binning: IrfBinning) -> None:
    np.savez(
        path,
        E_bins_edge=binning.energy,
        theta_bins_edge=binning.cos_theta,
        E_bins_center=binning.energy_centers,
        theta_bins_center=binning.cos_theta_centers,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e", "--exponent", type=float, default=0.0, help="spectral index of the input MC weights"
    )
    parser.add_argument(
        "--energy-binning",
        choices=("log", "giem"),
        default="log",
        help="log: N log bins between 1 GeV and 10 TeV; giem: GIEM energy bins",
    )
    parser.add_argument(
        "--n-energy-bins",
        type=int,
        default=N_ENERGY_BINS,
        help="number of energy bins for the log binning",
    )
    parser.add_argument("--n-cos-theta-bins", type=int, default=N_COS_THETA_BINS)


def run(args: argparse.Namespace, paths: Paths) -> None:
    binning = default_binning(args.energy_binning, args.n_energy_bins, args.n_cos_theta_bins)
    tag = Path("PSF") / f"exp{args.exponent:.1f}" / "script"
    results_dir = ensure_dir(paths.results_dir / tag)
    figures_dir = ensure_dir(paths.figures_dir / tag)
    plotting.use_style()

    files = browse_files(paths.filtered_mc_dir(args.exponent) / "*.root")
    if not files:
        raise FileNotFoundError(f"No ROOT files in {paths.filtered_mc_dir(args.exponent)}")
    log.info("Loading %d files", len(files))
    save_binning(results_dir / "infos_bins.npz", binning)

    df = preprocess(read_root_columns(files, TREE, INPUT_COLUMNS))
    log.info(summarize(df))
    df = compute_angular_error(df)
    plot_angular_error(df, figures_dir / f"angular_error_distribution_exp{args.exponent:.1f}.png")

    df, resolution, resolution_w = compute_resolution(df, binning)
    resolution.to_csv(results_dir / "resolution_per_bin.csv", index=False)
    resolution_w.to_csv(results_dir / "resolution_per_bin_w.csv", index=False)
    plot_resolution(
        resolution,
        binning,
        "HET Unweighted Angular Resolution Map",
        figures_dir / "resolutionMap_HET_unweighted.png",
    )
    plot_resolution(
        resolution_w,
        binning,
        "HET Weighted Angular Resolution Map",
        figures_dir / "resolutionMap_HET_weighted.png",
    )

    df["x_p"] = df["angular_error_degrees"] / df[RESOLUTION_COLUMN]
    params = fit_all_bins(df, binning, results_dir, figures_dir)
    param_path = results_dir / "fit_DK_params_all_bins.csv"
    params.to_csv(param_path, index=False)
    log.info("Fit parameters saved to %s", param_path)
