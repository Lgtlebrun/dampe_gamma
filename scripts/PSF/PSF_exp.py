import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import curve_fit
from scipy.stats import chi2

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

dirDir= "/home/users/l/lebrun/"
beegfs_path = "/dpnc/beegfs/dampe/tryf/ForKimberly/"
resPath = dirDir + "results/"

# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
PATH_GIEM_BINS = "/home/users/l/lebrun/results/all_bins_GIEM.npz"
# Load the effective area table
EFF_AREA_NPZ = np.load(PATH_EFFAREA)
EFF_AREA_TABLE = EFF_AREA_NPZ['effArea'] # Shape (n_E_bins, n_costheta_bins, n_phi_bins) ~ (20, 10, 360)

BINS_NPZ = np.load(PATH_GIEM_BINS)

# Get bins edges and centers
EAREA_E_BINS_EDGES = EFF_AREA_NPZ['E_bins_edge']
EAREA_E_BINS_CENTERS = np.sqrt(EAREA_E_BINS_EDGES[:-1] * EAREA_E_BINS_EDGES[1:])  # GeV

ENERGY_BINS_EDGES = BINS_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(ENERGY_BINS_EDGES[:-1] * ENERGY_BINS_EDGES[1:])  # GeV
COSTHETA_BINS_EDGES = EFF_AREA_NPZ['theta_bins_edge']
PHI_BINS_EDGES = EFF_AREA_NPZ['phi_bins_edge']

def load_obs_time(target):
    obsTime_path = beegfs_path + "obsTime/" + target + f"/obs_time_{target}.npz"
    obsTime = np.load(obsTime_path)
    
    hist_obsTime = obsTime['histogram']
    theta_bins_edge = obsTime['theta_edges']
    phi_bins_edge = obsTime['phi_edges']
    
    return hist_obsTime, theta_bins_edge, phi_bins_edge


# === Model functions ===
def king(xp, sigma, gamma):
    factor1 = 1 / (2 * np.pi * sigma * sigma)
    factor2 = (1 - 1 / gamma)
    factor3 = (1 + 0.5 * (1 / gamma) * ((xp / sigma) ** 2)) ** (-gamma)
    return factor1 * factor2 * factor3


def x_king(xp, sigma, gamma):
    return xp * king(xp, sigma, gamma)


def doubleKing(xp, A, F, sigma_c, gamma_c, sigma_t, gamma_t):
    """Full double King model."""
    return A * (F * x_king(xp, sigma_c, gamma_c) +
                (1 - F) * x_king(xp, sigma_t, gamma_t))


def chisq(data, model, error, dofs):
    residuals = data - model
    chi_squared = np.sum((residuals) ** 2 / (error ** 2))
    reduced_chi_squared = chi_squared / dofs
    p_value = 1 - chi2.cdf(chi_squared, dofs)
    return chi_squared, reduced_chi_squared, p_value


# === Stage 1: Fit Core ===
def core_model(xp, A_core, sigma_c, gamma_c):
    """Core King only (no tail)."""
    return A_core * x_king(xp, sigma_c, gamma_c)


# === Stage 2: Fit Tail ===
def tail_model(xp, A_tail, sigma_t, gamma_t):
    """Tail King only (no core)."""
    return A_tail * x_king(xp, sigma_t, gamma_t)

import warnings
def check_bounds_warnings(popt, bounds, param_names, stage_name):
    """
    Check if fitted parameters hit bounds and issue comprehensive warnings.
    
    Parameters:
    -----------
    popt : array-like
        Fitted parameters
    bounds : tuple
        (lower_bounds, upper_bounds) used in curve_fit
    param_names : list
        Names of parameters for clear warnings
    stage_name : str
        Name of fitting stage for context
    
    Returns:
    --------
    warnings_issued : list
        List of warning messages issued
    """
    lower_bounds, upper_bounds = bounds
    warnings_issued = []
    
    # More sensible tolerance criteria
    tolerance_abs = 1e-8  # Absolute tolerance for exact bound hits
    tolerance_rel = 0.05  # 5% relative tolerance based on parameter value
    
    for i, (param, lower, upper, name) in enumerate(zip(popt, lower_bounds, upper_bounds, param_names)):
        
        # Check lower bound
        if lower != -np.inf:
            # Use absolute tolerance for exact hits, relative tolerance for "close"
            if abs(param - lower) <= tolerance_abs:
                msg = f"⚠️  {stage_name}: Parameter '{name}' HIT LOWER BOUND exactly (value: {param:.6f}, bound: {lower})"
                warnings.warn(msg, UserWarning, stacklevel=3)
                warnings_issued.append(msg)
            elif param > 0 and param <= lower + tolerance_rel * abs(param):
                # Only warn if parameter is within 5% of its own value from the bound
                msg = f"⚠️  {stage_name}: Parameter '{name}' is very close to LOWER BOUND (value: {param:.6f}, bound: {lower})"
                warnings.warn(msg, UserWarning, stacklevel=3)
                warnings_issued.append(msg)
        
        # Check upper bound
        if upper != np.inf:
            # Use absolute tolerance for exact hits, relative tolerance for "close"
            if abs(param - upper) <= tolerance_abs:
                msg = f"⚠️  {stage_name}: Parameter '{name}' HIT UPPER BOUND exactly (value: {param:.6f}, bound: {upper})"
                warnings.warn(msg, UserWarning, stacklevel=3)
                warnings_issued.append(msg)
            elif param >= upper - tolerance_rel * abs(param):
                # Only warn if parameter is within 5% of its own value from the bound
                msg = f"⚠️  {stage_name}: Parameter '{name}' is very close to UPPER BOUND (value: {param:.6f}, bound: {upper})"
                warnings.warn(msg, UserWarning, stacklevel=3)
                warnings_issued.append(msg)
    
    return warnings_issued

def two_stage_fit(xp, data, error, percentile_cutoff=60):
    """
    Two-stage fitting procedure with comprehensive bound checking.
    
    Issues warnings when fitted parameters hit or approach bounds, which may indicate:
    - Poor initial guesses
    - Inappropriate bounds
    - Insufficient data quality
    - Model inadequacy
    """
    
    # ---- Stage 1: Fit the core ----
    core_cutoff = xp < np.percentile(xp, percentile_cutoff)  # Fit core using inner 60%
    xp_core = xp[core_cutoff]
    data_core = data[core_cutoff]
    error_core = error[core_cutoff]

    # Initial guess and bounds for core
    p0_core = [max(data_core), 0.5, 2.0]
    bounds_core = ([0, 0.01, 1.01], [np.inf, 3, 10])
    param_names_core = ['A_core', 'sigma_c', 'gamma_c']

    try:
        popt_core, pcov_core = curve_fit(
            core_model, xp_core, data_core, sigma=error_core, p0=p0_core, bounds=bounds_core, maxfev=10000
        )
        A_core, sigma_c, gamma_c = popt_core
        
        # Check bounds for core fit
        core_warnings = check_bounds_warnings(popt_core, bounds_core, param_names_core, "CORE FIT")
        
    except Exception as e:
        raise RuntimeError(f"Core fitting failed: {str(e)}")

    core_fit = core_model(xp, *popt_core)

    # ---- Stage 2: Fit the tail ----
    residuals = data - core_fit
    residuals[residuals < 0] = 0  # Remove negative residuals

    # Fit tail using outer 50%
    tail_cutoff = xp > np.percentile(xp, 100-percentile_cutoff)
    xp_tail = xp[tail_cutoff]
    residuals_tail = residuals[tail_cutoff]
    error_tail = error[tail_cutoff]

    p0_tail = [np.max(residuals_tail), 1.0, 2.0]
    bounds_tail = ([0, 0.1, 1.01], [np.inf, 10, 10])
    param_names_tail = ['A_tail', 'sigma_t', 'gamma_t']

    try:
        popt_tail, pcov_tail = curve_fit(
            tail_model, xp_tail, residuals_tail, sigma=error_tail, p0=p0_tail, bounds=bounds_tail
        )
        A_tail, sigma_t, gamma_t = popt_tail
        
        # Check bounds for tail fit
        tail_warnings = check_bounds_warnings(popt_tail, bounds_tail, param_names_tail, "TAIL FIT")
        
    except Exception as e:
        raise RuntimeError(f"Tail fitting failed: {str(e)}")

    tail_fit = tail_model(xp, *popt_tail)

    # ---- Combine Core + Tail ----
    A_total = A_core + A_tail
    F = A_core / A_total
    model_total = core_fit + tail_fit

    dofs = len(xp) - 6  # 6 parameters total
    chi2_val, chi2_red, p_val = chisq(data, model_total, error, dofs)
    
    # Collect all warnings
    all_warnings = core_warnings + tail_warnings if 'core_warnings' in locals() and 'tail_warnings' in locals() else []
    
    # Issue summary warning if any bounds were hit
    if all_warnings:
        summary_msg = f"🚨 FIT QUALITY WARNING: {len(all_warnings)} parameter(s) hit or approached bounds. This may indicate poor fit quality, inappropriate bounds, or model inadequacy. Consider:\n"
        summary_msg += "   • Adjusting parameter bounds\n"
        summary_msg += "   • Improving initial guesses\n"
        summary_msg += "   • Checking data quality\n"
        summary_msg += "   • Evaluating model appropriateness"
        warnings.warn(summary_msg, UserWarning, stacklevel=2)

    return {
        'params': {
            'A': A_total,
            'F': F,
            'sigma_c': sigma_c,
            'gamma_c': gamma_c,
            'sigma_t': sigma_t,
            'gamma_t': gamma_t
        },
        'cov_core': pcov_core,
        'cov_tail': pcov_tail,
        'chi2': chi2_val,
        'chi2_red': chi2_red,
        'p_value': p_val,
        'model_total': model_total,
        'core_fit': core_fit,
        'tail_fit': tail_fit,
        'warnings_issued': all_warnings  # New: list of all warnings for programmatic access
    }

if __name__ == "__main__":

    target = sys.argv[1] if len(sys.argv) > 1 else sys.exit("Please provide a target name as an argument.")

    verbose = True

    PSF_path = resPath + "PSF/exp0.0/script/fit_DK_params_all_bins.csv"
    PSF_df = pd.read_csv(PSF_path)
    if verbose:
        print(PSF_df.head())

    PSF_infos_path = resPath + "PSF/exp0.0/script/infos_bins.npz"
    PSF_infos = np.load(PSF_infos_path)

    if 'E_bins_edge' in PSF_infos:

        PSF_E_bins_edge = PSF_infos['E_bins_edge']
        PSF_theta_bins_edge = PSF_infos['theta_bins_edge']
        PSF_E_bins_center = PSF_infos['E_bins_center']
        PSF_theta_bins_center = PSF_infos['theta_bins_center']
        if verbose :
            print(f"E_bins_edge: {PSF_E_bins_edge}")
            print(f"theta_bins_edge: {PSF_theta_bins_edge}")
            print(f"E_bins_center: {PSF_E_bins_center}")
            print(f"theta_bins_center: {PSF_theta_bins_center}")

    obsTime, obsT_theta_bins_edge, obsT_phi_bins_edge = load_obs_time(target)

    