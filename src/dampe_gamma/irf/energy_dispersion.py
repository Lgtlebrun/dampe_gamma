"""Energy dispersion: relative energy deviation, resolution and dispersion fit per bin.

For each event the relative deviation Ediff = (E_reco - E_true) / E_true is computed.
Its spread per (energy, cos theta) bin gives the energy resolution S_D, and the scaled
deviation x_D = Ediff / S_D is fitted with a three piece dispersion function D(x_D).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_function
from scipy.stats import chi2 as chi2_distribution

from dampe_gamma import plotting
from dampe_gamma.binning import E_MAX_GEV, IrfBinning, cos_theta_edges, energy_edges
from dampe_gamma.config import Paths
from dampe_gamma.io import browse_files, ensure_dir, load_npz, read_root_columns, summarize
from dampe_gamma.stats import (
    COS_THETA_BIN_COLUMN,
    E_BIN_COLUMN,
    DensityHistogram,
    assign_bins,
    chisq,
    density_histogram,
    half_width_68,
    symmetric_containment_radius,
    table_to_grid,
)

log = logging.getLogger(__name__)

TREE = "MC"
INPUT_COLUMNS = ("tt_theta", "tt_ekin", "tt_bgoTotalEcorr_GeV", "weight")
N_ENERGY_BINS = 8
N_COS_THETA_BINS = 5
N_EDIFF_BINS = 100
N_XD_BINS = 100
CONTAINMENT_PERCENT = 68

MIN_COUNTS = 5
MIN_DENSITY = 1e-3
MAX_RELATIVE_ERROR = 0.5
N_RANDOM_STARTS = 30
N_WALK_STEPS = 30
WALK_NOISE_SCALE = 0.2
WALK_MAXFEV = 30000
D_PARAM_NAMES = (
    "x0",
    "xbar_l",
    "xbar_r",
    "sigma_l",
    "gamma_l",
    "sigma_i",
    "gamma_i",
    "sigma_r",
    "gamma_r",
    "N",
)
D_BOUNDS = (
    [-20, -15, -3, 0.001, 0.01, 1e-8, 0.01, 0.01, 0.4, 0],
    [20, 0, 1, 13, 10, 6, 10, 6, 10, 100],
)

RESOLUTION_COLUMNS = {
    "resolution": "half width of the 16 to 84 percentile interval",
    "resolution_w": "weighted half width of the 16 to 84 percentile interval",
    "resolution_r": "symmetric 68% containment radius around zero",
    "resolution_w_r": "weighted symmetric 68% containment radius around zero",
}


class EmptyBinError(Exception):
    """Raised when no histogram bin survives the quality mask."""


def asymmetric_gaussian(x: np.ndarray, sigma: float, k: float, b: float, p: float) -> np.ndarray:
    """Normalised generalised Gaussian of power p, peak b and asymmetry k."""
    factor_k = k / (1 + k**2)
    factor_p = p / (sigma * gamma_function(1 / p))
    shape = np.where(
        x - b >= 0,
        np.exp(-((k * np.abs(x - b) / sigma) ** p)),
        np.exp(-((np.abs(x - b) / (sigma * k)) ** p)),
    )
    return factor_k * factor_p * shape


def double_asymmetric_gaussian(
    x: np.ndarray,
    F: float,
    sigma1: float,
    k1: float,
    b1: float,
    p1: float,
    sigma2: float,
    k2: float,
    b2: float,
    p2: float,
) -> np.ndarray:
    """Mixture of two asymmetric Gaussians with weight F on the first."""
    return F * asymmetric_gaussian(x, sigma1, k1, b1, p1) + (1 - F) * asymmetric_gaussian(
        x, sigma2, k2, b2, p2
    )


def exp_power(x: np.ndarray, x0: float, sigma: float, gamma: float) -> np.ndarray:
    """Unnormalised exp(-|x - x0|^gamma / (2 sigma^gamma))."""
    return np.exp(-0.5 * np.abs((x - x0) / sigma) ** gamma)


def side_normalisations(
    x0: float,
    xbar_l: float,
    xbar_r: float,
    sigma_l: float,
    gamma_l: float,
    sigma_i: float,
    gamma_i: float,
    sigma_r: float,
    gamma_r: float,
    N: float,
) -> tuple[float, float]:
    """Normalisations of the left and right pieces of the dispersion function."""
    norm_l = N * exp_power(xbar_l, x0, sigma_i, gamma_i) / exp_power(xbar_l, x0, sigma_l, gamma_l)
    norm_r = N * exp_power(xbar_r, x0, sigma_i, gamma_i) / exp_power(xbar_r, x0, sigma_r, gamma_r)
    return norm_l, norm_r


def piece_masks(x: np.ndarray, x0: float, xbar_l: float, xbar_r: float):
    """Masks of the left, central and right pieces."""
    return x - x0 < xbar_l, (x - x0 >= xbar_l) & (x - x0 <= xbar_r), x - x0 > xbar_r


def dispersion_function(
    x: np.ndarray,
    x0: float,
    xbar_l: float,
    xbar_r: float,
    sigma_l: float,
    gamma_l: float,
    sigma_i: float,
    gamma_i: float,
    sigma_r: float,
    gamma_r: float,
    N: float,
) -> np.ndarray:
    """Three piece dispersion function D(x): left tail, core and right tail."""
    values = np.zeros_like(x)
    mask_l, mask_i, mask_r = piece_masks(x, x0, xbar_l, xbar_r)
    norm_l, norm_r = side_normalisations(
        x0, xbar_l, xbar_r, sigma_l, gamma_l, sigma_i, gamma_i, sigma_r, gamma_r, N
    )
    values[mask_l] = norm_l * exp_power(x[mask_l], x0, sigma_l, gamma_l)
    values[mask_i] = N * exp_power(x[mask_i], x0, sigma_i, gamma_i)
    values[mask_r] = norm_r * exp_power(x[mask_r], x0, sigma_r, gamma_r)
    return values


def random_walk_fit(
    model: Callable,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    p0: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    norm: np.ndarray,
    n_steps: int = N_WALK_STEPS,
    noise_scale: float = 0.1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Repeat curve_fit from perturbed starting points; keep the lowest chi2 in counts."""
    rng = np.random.default_rng() if rng is None else rng
    best_popt, best_fit_vals, best_chi2 = None, None, np.inf
    for _ in range(n_steps):
        start = []
        for p, lower, upper in zip(p0, *bounds, strict=True):
            if p != 0:
                perturbed = p * (1 + noise_scale * rng.uniform(-1, 1))
            else:
                perturbed = noise_scale * rng.standard_normal()
            start.append(np.clip(perturbed, lower, upper))
        try:
            popt, _ = curve_fit(
                model,
                x,
                y,
                p0=start,
                sigma=yerr,
                bounds=bounds,
                absolute_sigma=True,
                maxfev=WALK_MAXFEV,
            )
        except RuntimeError:
            continue
        fit_vals = model(x, *popt)
        chi2_val = chisq(y * norm, fit_vals * norm, yerr * norm, len(x) - len(popt))[0]
        if chi2_val < best_chi2:
            best_popt, best_fit_vals, best_chi2 = popt, fit_vals, chi2_val
    return best_popt, best_fit_vals, best_chi2


def random_start_fit(
    model: Callable,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    bounds: tuple[Sequence[float], Sequence[float]],
    norm: np.ndarray,
    n_starts: int = N_RANDOM_STARTS,
    noise_scale: float = 0.1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Random walk fits from starting points drawn uniformly within the bounds."""
    rng = np.random.default_rng() if rng is None else rng
    starts = [
        [
            rng.uniform(lower, upper)
            if np.isfinite(lower) and np.isfinite(upper)
            else rng.standard_normal()
            for lower, upper in zip(*bounds, strict=True)
        ]
        for _ in range(n_starts)
    ]
    best_popt, best_fit_vals, best_chi2 = None, None, np.inf
    for p0 in starts:
        try:
            popt, fit_vals, chi2_val = random_walk_fit(
                model, x, y, yerr, p0, bounds, norm, N_WALK_STEPS, noise_scale, rng
            )
        except (RuntimeError, ValueError) as err:
            log.debug("Random walk aborted: %s", err)
            continue
        if chi2_val < best_chi2:
            best_popt, best_fit_vals, best_chi2 = popt, fit_vals, chi2_val
            log.debug("New best chi2 = %.2f", chi2_val)
    return best_popt, best_fit_vals, best_chi2


@dataclass
class DispersionFit:
    """Result of the dispersion function fit on one x_D histogram."""

    params: np.ndarray
    chi2: float
    ndof: int
    p_value: float
    histogram: DensityHistogram


def xd_histogram(x_d: np.ndarray, n_bins: int = N_XD_BINS) -> DensityHistogram:
    """Normalised x_D histogram keeping bins with enough counts and small relative errors."""
    hist = density_histogram(x_d, n_bins)
    mask = (
        (hist.counts > MIN_COUNTS)
        & (hist.density > MIN_DENSITY)
        & (hist.errors < MAX_RELATIVE_ERROR * hist.density)
    )
    return hist.masked(mask)


def fit_xd_bin(
    x_d: np.ndarray,
    n_bins: int = N_XD_BINS,
    n_starts: int = N_RANDOM_STARTS,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> DispersionFit:
    """Fit the dispersion function to the x_D values of one bin."""
    if len(x_d) == 0:
        raise EmptyBinError("Empty bin")
    hist = xd_histogram(x_d, n_bins)
    if len(hist.centers) == 0:
        raise EmptyBinError("No bins remaining after masking. Fit cannot proceed.")

    popt, _, chi2_val = random_start_fit(
        dispersion_function,
        hist.centers,
        hist.density,
        hist.errors,
        D_BOUNDS,
        hist.widths * hist.total,
        n_starts=n_starts,
        noise_scale=WALK_NOISE_SCALE,
        rng=rng,
    )
    if popt is None:
        raise RuntimeError("No dispersion fit converged")
    ndof = len(hist.centers) - len(popt)
    p_value = 1 - chi2_distribution.cdf(chi2_val, ndof)
    return DispersionFit(popt, chi2_val, ndof, p_value, hist)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to E_true (GeV), E_reco, theta_True; add cos(theta) and the relative deviation."""
    df = df.rename(
        columns={"tt_theta": "theta_True", "tt_ekin": "E_true", "tt_bgoTotalEcorr_GeV": "E_reco"}
    )
    df["E_true"] = df["E_true"] * 1e-3
    df = df[df["E_true"] <= E_MAX_GEV].copy()
    df["cosTheta"] = np.cos(df["theta_True"])
    df["Ediff"] = (df["E_reco"] - df["E_true"]) / df["E_true"]
    return df


def compute_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """Energy resolution per bin with the four estimators of RESOLUTION_COLUMNS."""
    grouped = df.groupby([E_BIN_COLUMN, COS_THETA_BIN_COLUMN])[["Ediff", "weight"]]
    estimators = {
        "resolution": lambda g: half_width_68(g["Ediff"].values),
        "resolution_w": lambda g: half_width_68(g["Ediff"].values, g["weight"].values),
        "resolution_r": lambda g: symmetric_containment_radius(
            g["Ediff"].values, percent=CONTAINMENT_PERCENT
        ),
        "resolution_w_r": lambda g: symmetric_containment_radius(
            g["Ediff"].values, g["weight"].values, CONTAINMENT_PERCENT
        ),
    }
    return pd.DataFrame({name: grouped.apply(f) for name, f in estimators.items()}).reset_index()


def _bin_label(binning: IrfBinning, e_bin: int, cos_theta_bin: int) -> str:
    return (
        f"E: {binning.energy_centers[e_bin]:.1f} GeV, "
        f"cos(θ): {binning.cos_theta_centers[cos_theta_bin]:.2f}"
    )


def plot_ediff(
    df: pd.DataFrame, binning: IrfBinning, ediff_bins: np.ndarray, fig_dir: Path
) -> None:
    """Ediff distributions per bin, on one grid and one figure per bin."""
    n_energy, n_cos_theta = binning.shape
    groups = dict(list(df.groupby([E_BIN_COLUMN, COS_THETA_BIN_COLUMN])))

    fig, axes = plt.subplots(
        n_energy, n_cos_theta, figsize=(12, 10), sharex=True, sharey=True, squeeze=False
    )
    for (e_bin, cos_theta_bin), group in groups.items():
        ax = axes[e_bin, cos_theta_bin]
        ax.hist(group["Ediff"], bins=ediff_bins, density=True, alpha=0.7)
        ax.set_title(_bin_label(binning, e_bin, cos_theta_bin))
        ax.set_xlabel("E_reco - E_true / E_true")
        ax.set_ylabel("Density")
        ax.grid(True)
    fig.tight_layout()
    plotting.save(fig, fig_dir / "Ediff_per_E_cosTheta.png")

    for (e_bin, cos_theta_bin), group in groups.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(group["Ediff"], bins=ediff_bins, density=True, alpha=0.7)
        p16, p84 = np.percentile(group["Ediff"], [16, 84])
        ax.axvline(p16, color="red", linestyle="--", label="16th percentile")
        ax.axvline(p84, color="blue", linestyle="--", label="84th percentile")
        ax.arrow(
            p16,
            0.1,
            p84 - p16,
            0,
            head_width=0.02,
            head_length=0.02,
            fc="black",
            ec="black",
            length_includes_head=True,
        )
        ax.axvline(0, color="black", linestyle="-", label="E_reco = E_true")
        r = symmetric_containment_radius(group["Ediff"], group["weight"], CONTAINMENT_PERCENT)
        ax.annotate(
            "",
            xy=(r, 1),
            xytext=(-r, 1),
            arrowprops={"arrowstyle": "<->", "color": "purple", "lw": 2},
        )
        ax.legend()
        ax.set_title(_bin_label(binning, e_bin, cos_theta_bin))
        ax.set_xlabel("E_reco - E_true / E_true")
        ax.set_ylabel("Density")
        ax.grid(True)
        e_center = binning.energy_centers[e_bin]
        ct_center = binning.cos_theta_centers[cos_theta_bin]
        plotting.save(fig, fig_dir / f"Ediff_E{e_center:.1f}_cosTheta{ct_center:.2f}.png")


def _resolution_axis(fig, ax, grid: np.ndarray, binning: IrfBinning, title: str) -> None:
    mesh = ax.pcolormesh(binning.energy, binning.cos_theta, grid.T, shading="auto", cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("cos(θ)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax).set_label("Energy Resolution")


def plot_resolution_maps(grids: dict[str, np.ndarray], binning: IrfBinning, fig_dir: Path) -> None:
    """Resolution map, and weighted versus unweighted comparisons for both estimators."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _resolution_axis(fig, ax, grids["resolution"], binning, "HET Energy Resolution Map")
    fig.tight_layout()
    plotting.save(fig, fig_dir / "HET_Energy_Resolution_Map.png")

    comparisons = (
        (
            "resolution",
            "resolution_w",
            " (Unweighted, Percentiles)",
            " (Weighted, Percentiles)",
            "",
        ),
        ("resolution_r", "resolution_w_r", " (Unweighted)", " (Weighted)", "_r"),
    )
    for left, right, left_title, right_title, suffix in comparisons:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        _resolution_axis(fig, ax1, grids[left], binning, "HET Energy Resolution Map" + left_title)
        _resolution_axis(fig, ax2, grids[right], binning, "HET Energy Resolution Map" + right_title)
        fig.tight_layout()
        plotting.save(fig, fig_dir / f"HET_Energy_Resolution_Map_comparison{suffix}.png")


def plot_xd_fit(ax, fit: DispersionFit, e_bin: int, cos_theta_bin: int) -> None:
    """Data points, fitted D(x_D) and its three pieces."""
    hist = fit.histogram
    chi2_label = f"$\\chi^2$/ndof = {fit.chi2:.2f}/{fit.ndof} = {fit.chi2 / fit.ndof:.2f}"
    p_label = "p-value < 1e-16" if fit.p_value < 1e-16 else f"p-value = {fit.p_value:.2g}"
    ax.errorbar(
        hist.centers,
        hist.density,
        yerr=hist.errors,
        fmt="o",
        color="royalblue",
        alpha=0.7,
        label="Data",
        markersize=3,
    )
    ax.set_title(f"E_bin={e_bin}, θ_bin={cos_theta_bin}", fontsize=9)

    x0, xbar_l, xbar_r, sigma_l, gamma_l, sigma_i, gamma_i, sigma_r, gamma_r, N = fit.params
    norm_l, norm_r = side_normalisations(*fit.params)
    x_fit = np.linspace(np.min(hist.centers), np.max(hist.centers), 1000)
    ax.plot(
        x_fit,
        dispersion_function(x_fit, *fit.params),
        color="darkred",
        lw=2,
        label=f"Fit\n{chi2_label}\n{p_label}",
    )
    mask_l, mask_i, mask_r = piece_masks(x_fit, x0, xbar_l, xbar_r)
    ax.plot(
        x_fit[mask_l],
        norm_l * exp_power(x_fit[mask_l], x0, sigma_l, gamma_l),
        color="green",
        label="Left",
    )
    ax.plot(
        x_fit[mask_i],
        N * exp_power(x_fit[mask_i], x0, sigma_i, gamma_i),
        color="blue",
        label="Core",
    )
    ax.plot(
        x_fit[mask_r],
        norm_r * exp_power(x_fit[mask_r], x0, sigma_r, gamma_r),
        color="orange",
        label="Right",
    )
    ax.set_xlabel("$x_D$")
    ax.set_ylabel("$D(x_D)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2)
    ax.grid()
    ax.legend()


def fit_all_bins(
    df: pd.DataFrame,
    binning: IrfBinning,
    results_dir: Path,
    fig_dir: Path,
    n_starts: int = N_RANDOM_STARTS,
    rng: np.random.Generator | None = None,
) -> dict[tuple[int, int], DispersionFit]:
    """Fit D(x_D) in every bin, save the parameters per bin and the overview figure."""
    n_energy, n_cos_theta = binning.shape
    fig, axes = plt.subplots(
        n_energy,
        n_cos_theta,
        figsize=(n_cos_theta * 4, n_energy * 3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle("Scaled Energy Deviation x_D in Each (E_bin, θ_bin)", fontsize=16)

    fits = {}
    for e_bin in range(n_energy):
        for cos_theta_bin in range(n_cos_theta):
            log.info("Processing E_bin=%d, cos_theta_bin=%d", e_bin, cos_theta_bin)
            in_bin = (df[E_BIN_COLUMN] == e_bin) & (df[COS_THETA_BIN_COLUMN] == cos_theta_bin)
            ax = axes[e_bin, cos_theta_bin]
            try:
                fit = fit_xd_bin(df.loc[in_bin, "x_D"].to_numpy(), n_starts=n_starts, rng=rng)
            except (EmptyBinError, RuntimeError) as err:
                log.warning(
                    "Fit failed for E_bin=%d, cos_theta_bin=%d: %s", e_bin, cos_theta_bin, err
                )
                ax.set_visible(False)
                continue
            np.save(results_dir / f"fit_params_LET_E{e_bin}_theta{cos_theta_bin}.npy", fit.params)
            log.info("chi2/ndof = %.2f/%d, p value %.3g", fit.chi2, fit.ndof, fit.p_value)
            fits[(e_bin, cos_theta_bin)] = fit
            plot_xd_fit(ax, fit, e_bin, cos_theta_bin)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plotting.save(fig, fig_dir / "xD_allBins_fits.png")
    return fits


def load_binning(
    energy_binning: str, paths: Paths, n_energy_bins: int, n_cos_theta_bins: int
) -> IrfBinning:
    """Energy edges from the effective area table (default), log or GIEM; linear cos(theta)."""
    if energy_binning == "effective-area":
        energy = load_npz(paths.effective_area_file)["E_bins_edge"]
    else:
        energy = energy_edges(energy_binning, n_energy_bins)
    return IrfBinning(energy, cos_theta_edges(n_cos_theta_bins))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e", "--exponent", type=float, default=0.0, help="spectral index of the input MC weights"
    )
    parser.add_argument(
        "--energy-binning",
        choices=("effective-area", "log", "giem"),
        default="effective-area",
        help="energy edges of the effective area table, log bins or GIEM bins",
    )
    parser.add_argument(
        "--n-energy-bins",
        type=int,
        default=N_ENERGY_BINS,
        help="number of energy bins for the log binning",
    )
    parser.add_argument("--n-cos-theta-bins", type=int, default=N_COS_THETA_BINS)
    parser.add_argument(
        "--n-starts",
        type=int,
        default=N_RANDOM_STARTS,
        help="random starting points of the dispersion fit per bin",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed of the fit random walk")


def run(args: argparse.Namespace, paths: Paths) -> None:
    binning = load_binning(args.energy_binning, paths, args.n_energy_bins, args.n_cos_theta_bins)
    tag = Path("Edist") / f"exp{args.exponent:.1f}"
    results_dir = ensure_dir(paths.results_dir / tag)
    fig_dir = ensure_dir(paths.figures_dir / tag)
    plotting.use_style()

    files = browse_files(paths.filtered_mc_dir(args.exponent) / "*.root")
    if not files:
        raise FileNotFoundError(f"No ROOT files in {paths.filtered_mc_dir(args.exponent)}")
    df = preprocess(read_root_columns(files, TREE, INPUT_COLUMNS))
    log.info(summarize(df))

    ediff_bins = np.linspace(df["Ediff"].min(), df["Ediff"].max(), N_EDIFF_BINS + 1)
    df = assign_bins(df, binning)
    plot_ediff(df, binning, ediff_bins, fig_dir)

    resolution = compute_resolution(df)
    grids = {name: table_to_grid(resolution, binning.shape, name) for name in RESOLUTION_COLUMNS}
    plot_resolution_maps(grids, binning, fig_dir)
    np.savez_compressed(
        results_dir / "energy_resolution_grid_LET.npz",
        resolution_grid=grids["resolution"],
        resolution_grid_w=grids["resolution_w"],
        resolution_grid_r=grids["resolution_r"],
        resolution_grid_w_r=grids["resolution_w_r"],
        E_bins_edge=binning.energy,
        theta_bins_edge=binning.cos_theta,
    )

    df = df.merge(
        resolution[[E_BIN_COLUMN, COS_THETA_BIN_COLUMN, "resolution"]],
        on=[E_BIN_COLUMN, COS_THETA_BIN_COLUMN],
        how="left",
    )
    df["x_D"] = df["Ediff"] / df["resolution"]
    fit_all_bins(df, binning, results_dir, fig_dir, args.n_starts, np.random.default_rng(args.seed))
