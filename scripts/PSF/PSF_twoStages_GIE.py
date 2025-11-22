import ROOT, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

RESULT_PATH = "/home/users/l/lebrun/results/"

# Load the npz file
data = np.load(RESULT_PATH + "E_bin_GIEM.npz")
# Access the array by the key you used when saving
E_bins_edge = data["E_bins_edge"]

E_bins_center = np.sqrt(E_bins_edge[:-1] * E_bins_edge[1:])
n_E_bins = len(E_bins_center)


print("E_bins :", E_bins_center)
print("E_bins_edge :", E_bins_edge)

cos_theta_min = 0.5
cos_theta_max = 1
n_theta_bins = 5
theta_bins_edge = np.linspace(cos_theta_min, cos_theta_max, n_theta_bins + 1)
print("theta_bins_edge :", theta_bins_edge)
theta_bins_center = (theta_bins_edge[:-1] + theta_bins_edge[1:])/2

infosPath = RESULT_PATH + "PSF/exp0.0/script/infos_bins.npz"
np.savez(infosPath, E_bins_edge=E_bins_edge, theta_bins_edge=theta_bins_edge,
         E_bins_center=E_bins_center, theta_bins_center=theta_bins_center)
print(f"Saved bin edges and centers to {infosPath}")
# Setup colormap
cmap = plt.get_cmap('viridis').copy()
cmap.set_under('k', 1e-6)

def browse_files(filenames):
    """
    Browse through the list of filenames and return a sorted list of files.
    """
    if isinstance(filenames, str):
        files = sorted(glob.glob(filenames))
    elif isinstance(filenames, list):
        files = []
        for filename in filenames:
            files += sorted(glob.glob(filename))
    else:
        raise ValueError("Input should be a string or a list of strings.")
    
    return files

def preprocess_df(df:pd.DataFrame):

    df['E_true'] = df['tt_ekin'] * 1e-3

    df = df.rename(columns={'tt_theta' : 'theta_True', 'tt_phi' : 'phi_True'})

    df = df[df['E_true'] <= 1e4]

    df['theta_True_deg']= np.rad2deg(df['theta_True'])
    df['cosTheta'] = np.cos(df['theta_True'])

    df['phi_True_deg']= np.rad2deg(df['phi_True'])
    df['theta_BestSTK_deg']= df['theta_BestSTK']
    df['phi_BestSTK_deg']= df['phi_BestSTK']

    df['theta_BestSTK'] = np.deg2rad(df['theta_BestSTK'])
    df['phi_BestSTK'] = np.deg2rad(df['phi_BestSTK'])

    df = df.loc[(df['theta_True'] > 1e-7) & (df['theta_BestSTK'] > 1e-7)]

    return df

def infos_df(df, df_name="DataFrame"):
    """
    Print comprehensive information about a DataFrame including statistics
    for energy, theta, and event counts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    df_name : str, optional
        Name identifier for the DataFrame (default: "DataFrame")
    """
    
    print(f"\n=== {df_name} Information ===")
    
    # Basic info
    print(f"Number of entries: {len(df)}")
    
    if len(df) == 0:
        print("DataFrame is empty!")
        return
    
    # Energy information (if E_true column exists)
    if 'E_true' in df.columns:
        print(f"Minimum E_true: {df['E_true'].min():.6f} GeV")
        print(f"Maximum E_true: {df['E_true'].max():.6f} GeV")
        print(f"Mean E_true: {df['E_true'].mean():.6f} GeV")
        print(f"Median E_true: {df['E_true'].median():.6f} GeV")
    
    # Theta information (check for various theta column names)
    theta_cols = ['theta_True', 'tt_theta', 'theta_True_deg']
    theta_col = None
    
    for col in theta_cols:
        if col in df.columns:
            theta_col = col
            break
    
    if theta_col:
        if 'deg' in theta_col:
            # Already in degrees
            theta_deg = df[theta_col]
            print(f"Minimum {theta_col}: {theta_deg.min():.2f} degrees")
            print(f"Maximum {theta_col}: {theta_deg.max():.2f} degrees")
            print(f"Mean {theta_col}: {theta_deg.mean():.2f} degrees")
        else:
            # Convert from radians to degrees
            theta_deg = np.rad2deg(df[theta_col])
            print(f"Minimum {theta_col}: {theta_deg.min():.2f} degrees ({df[theta_col].min():.4f} rad)")
            print(f"Maximum {theta_col}: {theta_deg.max():.2f} degrees ({df[theta_col].max():.4f} rad)")
            print(f"Mean {theta_col}: {theta_deg.mean():.2f} degrees ({df[theta_col].mean():.4f} rad)")
    
    phi_cols = ['phi_True', 'tt_phi', 'phi_True_deg']
    phi_col = None
    
    for col in phi_cols:
        if col in df.columns:
            phi_col = col
            break
    
    if phi_col:
        if 'deg' in phi_col:
            # Already in degrees
            phi_deg = df[phi_col]
            print(f"Minimum {phi_col}: {phi_deg.min():.2f} degrees")
            print(f"Maximum {phi_col}: {phi_deg.max():.2f} degrees")
            print(f"Mean {phi_col}: {phi_deg.mean():.2f} degrees")
        else:
            # Convert from radians to degrees
            phi_deg = np.rad2deg(df[phi_col])
            print(f"Minimum {phi_col}: {phi_deg.min():.2f} degrees ({df[phi_col].min():.4f} rad)")
            print(f"Maximum {phi_col}: {phi_deg.max():.2f} degrees ({df[phi_col].max():.4f} rad)")
            print(f"Mean {phi_col}: {phi_deg.mean():.2f} degrees ({df[phi_col].mean():.4f} rad)")
    

    # HET information (if tt_isHET column exists)
    if 'tt_isHET' in df.columns:
        het_events = df[df['tt_isHET'] == 1]
        print(f"Number of HET events (tt_isHET == 1): {len(het_events)}")
        print(f"HET fraction: {len(het_events)/len(df)*100:.2f}%")
    
    # cosTheta information (if exists)
    if 'cosTheta' in df.columns:
        print(f"Minimum cosTheta: {df['cosTheta'].min():.4f}")
        print(f"Maximum cosTheta: {df['cosTheta'].max():.4f}")
        print(f"Mean cosTheta: {df['cosTheta'].mean():.4f}")
    
    # Weight information (if exists)
    if 'weight' in df.columns:
        print(f"Weight statistics:")
        print(f"  Sum of weights: {df['weight'].sum():.2e}")
        print(f"  Mean weight: {df['weight'].mean():.2e}")
        print(f"  Min weight: {df['weight'].min():.2e}")
        print(f"  Max weight: {df['weight'].max():.2e}")
    
    # Additional energy cuts information
    if 'E_true' in df.columns:
        high_energy_mask = df['E_true'] <= 1e4  # 10 TeV
        print(f"Number of events with E_true <= 10 TeV: {high_energy_mask.sum()}")
        
        # Common energy ranges
        gev_ranges = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
        for low, high in gev_ranges:
            mask = (df['E_true'] >= low) & (df['E_true'] < high)
            count = mask.sum()
            print(f"  Events in [{low}-{high}) GeV: {count}")
    
    # Additional theta cuts information
    if theta_col and 'deg' not in theta_col:
        theta_deg = np.rad2deg(df[theta_col])
        theta_60_mask = theta_deg < 60
        print(f"Number of events with theta < 60°: {theta_60_mask.sum()}")
        
        # Common theta ranges
        theta_ranges = [(0, 30), (30, 60), (60, 90), (90, 180)]
        for low, high in theta_ranges:
            mask = (theta_deg >= low) & (theta_deg < high)
            count = mask.sum()
            print(f"  Events in [{low}-{high})°: {count}")
    elif 'theta_True_deg' in df.columns:
        theta_60_mask = df['theta_True_deg'] < 60
        print(f"Number of events with theta < 60°: {theta_60_mask.sum()}")
    
    # Data types and memory usage
    print(f"\nColumn information:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        print(f"  {col}: {dtype}, {null_count} nulls")
    
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("=" * 50)

    # Additional info for angular columns and energy
    if 'theta_BestSTK' in df.columns:
        print(f"Theta_BestSTK ranges from {df['theta_BestSTK'].min()} to {df['theta_BestSTK'].max()} rad")
    if 'phi_BestSTK' in df.columns:
        print(f"Phi_BestSTK ranges from {df['phi_BestSTK'].min()} to {df['phi_BestSTK'].max()} rad")
    if 'phi_True' in df.columns:
        print(f"phi_True ranges from {df['phi_True'].min()} to {df['phi_True'].max()} rad")
    if 'theta_True' in df.columns:
        print(f"Theta_True ranges from {df['theta_True'].min()} to {df['theta_True'].max()} rad")
    if 'E_true' in df.columns:
        print(f"Energy ranges from {df['E_true'].min()} to {df['E_true'].max()} GeV")

def compute_angular_error(df):
    def angular_error(theta_true, phi_true, theta_reco, phi_reco):

        # Convert spherical to Cartesian coordinates
        def sph2cart(theta, phi):
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            return np.stack((x, y, z), axis=-1)

        v_true = sph2cart(theta_true, phi_true)
        v_reco = sph2cart(theta_reco, phi_reco)

        # # Compute dot product and then angle between vectors
        dot = np.sum(v_true * v_reco, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        mask = np.abs(dot) > 1
        if np.any(mask):
            print(f"ERROR: Found {np.sum(mask)} dot products not in [-1, 1]")
            print("Examples:", dot[mask][:5])  # print first 5 bad values

        angle_rad = np.arccos(dot)

        # delta_v = v_true - v_reco
        # delta_v_norm = np.linalg.norm(delta_v, axis=-1)

        return angle_rad

    df['angular_error'] = angular_error(df['theta_True'], df['phi_True'], df['theta_BestSTK'], df['phi_BestSTK'])

    df['angular_error_degrees'] = df['angular_error'] * 180 / np.pi

    # Count outliers
    # outliers = df[(df['angular_error_degrees'] > 10)]
    # print("Outliers count:", outliers.shape[0])
    # print("Outliers proportion : ", outliers.shape[0] / df.shape[0] *100, "%")

    # df = df.loc[(df['angular_error_degrees'] < 10)]
    return df

def error_xyz(theta_true, phi_true, theta_reco, phi_reco):
    # Convert spherical to Cartesian coordinates
    def sph2cart(theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack((x, y, z), axis=-1)

    v_true = sph2cart(theta_true, phi_true)
    v_reco = sph2cart(theta_reco, phi_reco)

    # Compute the difference in Cartesian coordinates
    delta_v = v_true - v_reco

    return delta_v

def plot_angular_error(df, figName):
    fig, ax = plt.subplots()

    range_ = (0, 10)
    column = 'angular_error_degrees'

    # --- Unweighted histogram ---
    ax.hist(df[column], bins=50, range=range_, histtype='step', color='green', label='Unweighted')
    ax.hist(df[column], bins=50, range=range_, weights=df['weight'],
                histtype='step', color='blue', label='Weighted')
    ax.set_title('Angular Error Distribution')
    ax.set_xlabel('Angular Error (degrees)')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)


    plt.tight_layout()
    fig.savefig(figName, dpi=400)

    # --- Figure 2: Normalized PSF (Probability Density Function) ---
    # Use log bins for angular error
    bins = np.logspace(np.log10(0.01), np.log10(10), 50)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    # Histogram values
    hist_unweighted, _ = np.histogram(df[column], bins=bins)
    hist_weighted, _ = np.histogram(df[column], bins=bins, weights=df['weight'])

    # Normalize to get density (probability per bin width)
    bin_widths = np.diff(bins)
    density = hist_unweighted / np.sum(hist_unweighted) / bin_widths
    density_w = hist_weighted / np.sum(hist_weighted) / bin_widths

    fig2 = plt.figure(figsize=(8,6))
    plt.plot(bin_centers, density, drawstyle='steps-mid', color='blue', label='Unweighted')
    plt.plot(bin_centers, density_w, drawstyle='steps-mid', color='green', label='Weighted')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Probability Density (per degree)')
    plt.title('Normalized PSF: Angular Error Distribution')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figName[:-4] + "_density.png", dpi=400)

def weighted_percentile(data, weights, percentiles):
    """Compute weighted percentiles."""
    data, weights = np.asarray(data), np.asarray(weights)
    percentiles = np.asarray(percentiles, dtype=float)

    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]

    cumulative_weight = np.cumsum(weights_sorted)
    total_weight = cumulative_weight[-1]

    if total_weight == 0:
        raise ValueError("Sum of weights is zero.")

    return np.interp(percentiles / 100 * total_weight, cumulative_weight, data_sorted)


def weighted_angular_resolution_68(group):
    angles = group['angular_error_degrees'].values
    weights = group['weight'].values

    # Alternative: 68% containment radius
    return weighted_percentile(angles, weights, 68)

    # Or if you want half-width of [16, 84]
    # lower, upper = weighted_percentile(angles, weights, [16, 84])
    # return (upper - lower) / 2


def angular_resolution_68(x):
    # Alternative: 68% containment radius
    return np.percentile(x, 68)

    # Or if you want half-width of [16, 84]
    # lower = np.percentile(x, 16)
    # upper = np.percentile(x, 84)
    # return (upper - lower) / 2


def compute_resolution(df): 
    df = df.copy()
    df['E_bin'] = pd.cut(df['E_true'], bins=E_bins_edge, labels=False, include_lowest=True)
    df['theta_bin'] = pd.cut(df['cosTheta'], bins=theta_bins_edge, labels=False, include_lowest=True)

    df = df.dropna(subset=['E_bin', 'theta_bin']).copy()
    df['E_bin'] = df['E_bin'].astype(int)
    df['theta_bin'] = df['theta_bin'].astype(int)

    grouped = df.groupby(['E_bin', 'theta_bin'])

    resolution_per_bin = grouped['angular_error_degrees'].apply(
        angular_resolution_68
    ).reset_index(name='angular_resolution')
    print(resolution_per_bin)

    resolution_per_bin_w = grouped.apply(
        weighted_angular_resolution_68
    ).reset_index(name='angular_resolution')
    print(resolution_per_bin_w)

    df = df.merge(resolution_per_bin_w, on=['E_bin', 'theta_bin'], how='left')

    return df, resolution_per_bin, resolution_per_bin_w

def save_df_to_csv(df : pd.DataFrame, fileName) :
    # Save weighted angular resolution per bin to CSV
    df.to_csv(fileName, index=False)

def plot_resolution(resolution_per_bin, figName):

    resolution_grid = np.full((n_E_bins, n_theta_bins), np.nan)
    for _, row in resolution_per_bin.iterrows():
        resolution_grid[int(row['E_bin']), int(row['theta_bin'])] = row['angular_resolution']


    # Prepare edges for pcolormesh
    E_edges = E_bins_edge

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(E_edges, theta_bins_edge, resolution_grid.T, shading='auto', cmap='viridis')

    # Log scale for energy
    ax.set_xscale('log')
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("cos(θ)")
    if "unweighted" in figName.lower():
        ax.set_title("HET Unweighted Angular Resolution Map")
    else:
        ax.set_title("HET Weighted Angular Resolution Map")
    # Colorbar
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Angular Resolution (degrees)")

    plt.tight_layout()
    fig.savefig(figName)


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
    return 2 * np.pi * A * (F * x_king(xp, sigma_c, gamma_c) +
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
    return 2 * np.pi * A_core * x_king(xp, sigma_c, gamma_c)


# === Stage 2: Fit Tail ===
def tail_model(xp, A_tail, sigma_t, gamma_t):
    """Tail King only (no core)."""
    return 2 * np.pi * A_tail * x_king(xp, sigma_t, gamma_t)

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
    bounds_core = ([0, 0.01, 0.1], [np.inf, 15, 50])
    param_names_core = ['A_core', 'sigma_c', 'gamma_c']

    try:
        popt_core, pcov_core = curve_fit(
            core_model, xp_core, data_core, sigma=error_core, p0=p0_core, bounds=bounds_core, maxfev=30000
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
    bounds_tail = ([0, 0.001, 0.1], [np.inf, 50, 50])
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

def dynamic_full_p0_bounds(x, y):
    """
    Dynamic initial guesses and bounds for full double King fit.
    """
    y_max = np.max(y)
    x_max = np.max(x)

    # Core: A, sigma, gamma
    A_c0 = 0.6 * y_max
    sigma_c0 = x_max / 5
    gamma_c0 = 2.0

    # Tail: A, sigma, gamma
    A_t0 = 0.4 * y_max
    sigma_t0 = x_max / 2
    gamma_t0 = 2.0

    p0 = [A_c0, sigma_c0, gamma_c0, A_t0, sigma_t0, gamma_t0]
    bounds_lower = [0, 0.01, 0.5, 0, 0.01, 0.5]
    bounds_upper = [10 * y_max, x_max, 15, 10 * y_max, 2 * x_max, 15]

    return p0, (bounds_lower, bounds_upper)


def doubleKing_fit(xp, A_c, sigma_c, gamma_c, A_t, sigma_t, gamma_t):
    A = A_c + A_t
    F  = A_c/A
    return doubleKing(xp, A, F, sigma_c, gamma_c, sigma_t, gamma_t)

def one_stage_full_fit(xp, data, error):
    """
    Fit all 6 parameters of the double King model at once on all data.
    """
    p0, bounds = dynamic_full_p0_bounds(xp, data)
    param_names = ['A_c', 'sigma_c', 'gamma_c', 'A_t', 'sigma_t', 'gamma_t']
    
    try:
        popt, pcov = curve_fit(
            doubleKing_fit, xp, data, sigma=error,
            p0=p0, bounds=bounds, maxfev=100000
        )
        warnings_full = check_bounds_warnings(popt, bounds, param_names, "FULL DOUBLE KING FIT")
    except Exception as e:
        raise RuntimeError(f"Full double King fit failed: {e}")

    model = doubleKing_fit(xp, *popt)
    dofs = len(xp) - len(popt)
    chi2_val, chi2_red, p_val = chisq(data, model, error, dofs)

    A_c, sigma_c, gamma_c, A_t, sigma_t, gamma_t = popt
    A_total = A_c + A_t
    F = A_c / A_total

    return {
        'params': {
            'A': A_total, 'F': F, 'A_c': A_c, 'sigma_c': sigma_c, 'gamma_c': gamma_c,
            'A_t': A_t, 'sigma_t': sigma_t, 'gamma_t': gamma_t
        },
        'cov': pcov,
        'chi2': chi2_val, 'chi2_red': chi2_red, 'p_value': p_val,
        'model_total': model,
        'warnings_issued': warnings_full
    }

def fit_bin_two_stage(subset, bins, fallback_threshold=2.5):
    """
    Bin data and perform two-stage fit with one-stage full fallback if χ²_red poor.
    """
    counts, bin_edges = np.histogram(subset["x_p"], bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    total = counts.sum()
    counts_norm = counts / (total * bin_widths)
    errors_norm = np.sqrt(counts) / (total * bin_widths)

    mask = (counts > 0) & (bin_centers > 1e-2)
    bin_centers, counts_norm, errors_norm = bin_centers[mask], counts_norm[mask], errors_norm[mask]

    percentiles = np.linspace(20, 80, 13)
    best_result, best_chi2_red = None, np.inf
    exception_record = None

    # Try two-stage fits across percentiles
    for percentile in percentiles:
        try:
            result = two_stage_fit(bin_centers, counts_norm, errors_norm, percentile_cutoff=percentile)
            chi2_red = result['chi2_red']
            if np.abs(chi2_red - 1) < np.abs(best_chi2_red - 1):
                best_result, best_chi2_red = result, chi2_red
        except Exception as e:
            print(f"Two-stage fit failed at percentile {percentile}: {e}")
            exception_record = e
            continue

    # Fallback to one-stage full 6-parameter fit if needed
    if best_result is None or best_chi2_red > fallback_threshold:
        print("⚠️ Two-stage fit unsatisfactory, attempting full double King fit...")
        try:
            full_result = one_stage_full_fit(bin_centers, counts_norm, errors_norm)
            print(f"Full fit χ²_red = {full_result['chi2_red']:.3f}")
            if full_result['chi2_red'] < best_chi2_red:
                print("✅ Full double King fit selected (better reduced χ²).")

                # Build core_fit and tail_fit from full fit parameters
                A_c, sigma_c, gamma_c = full_result['params']['A_c'], full_result['params']['sigma_c'], full_result['params']['gamma_c']
                A_t, sigma_t, gamma_t = full_result['params']['A_t'], full_result['params']['sigma_t'], full_result['params']['gamma_t']

                core_fit = core_model(bin_centers, A_c, sigma_c, gamma_c)
                tail_fit = core_model(bin_centers, A_t, sigma_t, gamma_t)

                # Populate core/tail info for plotting
                full_result.update({
                    'core_fit': core_fit,
                    'tail_fit': tail_fit,
                    'core_params': (A_c, sigma_c, gamma_c),
                    'tail_params': (A_t, sigma_t, gamma_t)
                })

                best_result = full_result
        except Exception as e:
            print(f"Full double King fit also failed: {e}")
            if best_result is None:
                raise exception_record

    best_result.update({
        'bin_centers': bin_centers,
        'counts_norm': counts_norm,
        'errors_norm': errors_norm
    })
    return best_result


# Define bounds
lower_bounds = [0, 0.5, 0, 1.00001, 0, 1.0001]
upper_bounds = [10, 1, 10, 10, 10, 10]

bounds = (lower_bounds, upper_bounds)

class EmptyBinException(Exception):
    """Custom exception for empty bins."""
    pass

def get_histogram(subset, bins):
    
    counts, bin_edges = np.histogram(subset['x_p'], bins=bins)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts)

    return bin_centers, counts, errors

def get_histogram_normed(subset, bins):

    bin_centers, counts, errors = get_histogram(subset, bins)
    # Normalize counts and errors
    total = np.sum(counts)
    print(f"Total number of counts in subset : {total}")
    counts_norm = counts / (bin_centers * total) 
    errors_norm = errors / (bin_centers * total)

    return bin_centers, counts, errors, counts_norm, errors_norm


def fit_xp_bin(subset, bins, bin_threshold=5) :
    
    bin_centers, counts, errors, counts_norm, errors_norm = get_histogram_normed(subset, bins)

    epsilon = 1e-5
    valid = (counts > bin_threshold)

    bin_centers_valid = bin_centers[valid]
    counts_norm_valid = counts_norm[valid]
    errors_norm_valid = errors_norm[valid]

    if len(bin_centers_valid) == 0 :
        raise EmptyBinException
    
    # ---- Initial guesses ----
    # Default fallback values
    p0_default = [0.2, 0.8, 0.5, 1.2, 0.5, 1.2]

    # Check if there's enough data to compute mean/std
    if len(subset["x_p"]) > 1 and np.sum(counts_norm_valid) > 0:
        try:
            
            std_dev = np.std(subset["x_p"])
            F_init = 0.8
            A_init = np.max(counts_norm_valid)
            p0 = [A_init, F_init, std_dev, 1.2, std_dev * 1.5, 1.2]

            # Clip to bounds
            p0 = np.clip(p0, lower_bounds, upper_bounds)

            if not np.isfinite(std_dev) or std_dev <= 0:
                raise ValueError("Invalid std_dev")

        except Exception as e:
            print(f"⚠️ Warning: falling back to default p0 due to: {e}")
            p0 = p0_default

    else:
        print("⚠️ Warning: empty or low-stat bin — using default p0")
        p0 = p0_default

    popt, pcov = curve_fit(
            doubleKing,
            bin_centers_valid,
            counts_norm_valid,
            p0=p0,
            sigma=errors_norm_valid,
            bounds=bounds,
            absolute_sigma=True,
            maxfev=20000
        )

    fit_vals = doubleKing(bin_centers_valid, *popt)
    # Compute chi-squared
    ndof = len(bin_centers_valid) - len(popt)  # number of degrees of freedom
    chi_squared, p_val = chisq(counts_norm_valid, fit_vals, errors_norm_valid, ndof)
    
    return popt, chi_squared, p_val, bin_centers_valid, counts_norm_valid, errors_norm_valid

def plot_fit_result(ax, subset, bins, fit_result, show_legend=True, chi2_ndof_label=""):
    """
    Plot the binned data with the full double King fit result.
    """
    if fit_result is None:
        counts, bin_edges = np.histogram(subset["x_p"], bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        total = counts.sum()
        counts_norm = counts / (total * bin_widths)
        errors_norm = np.sqrt(counts) / (total * bin_widths)  # Poisson errors normalized

        # Remove zero-count bins to avoid log(0) issues
        mask = (counts > 0) & (bin_centers > 1e-2)
        bin_centers = bin_centers[mask]
        counts_norm = counts_norm[mask]
        errors_norm = errors_norm[mask]
        ax.errorbar(bin_centers, counts_norm, yerr=errors_norm, fmt='o', label='Data')
        ax.text(0.1, 0.9, "Fit Failed", ha="center", va="center", color="red")
        return

    bin_centers = fit_result['bin_centers']
    counts_norm = fit_result['counts_norm']
    errors_norm = fit_result['errors_norm']
    model_total = fit_result['model_total']
    core_fit = fit_result['core_fit']
    tail_fit = fit_result['tail_fit']

    ax.errorbar(bin_centers, counts_norm, yerr=errors_norm, fmt='o', label='Data')
    ax.plot(bin_centers, core_fit, 'r--', label='Core Fit')
    ax.plot(bin_centers, tail_fit, 'b--', label='Tail Fit')
    ax.plot(bin_centers, model_total, 'g-', label='Total Fit')

    if show_legend:
        ax.legend(fontsize=8, loc='best')

    if chi2_ndof_label:
        ax.text(0.05, 0.95, chi2_ndof_label, transform=ax.transAxes,
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
def fit_all_bins(df, resPath, paramPath=None):
    """
    Fit the double King distribution to binned data in (E_bin, θ_bin) space using a two-stage fit.
    """
    fig, axes = plt.subplots(n_E_bins, n_theta_bins, figsize=(n_E_bins * 4, n_theta_bins * 3), sharex=True, sharey=True)
    fig.suptitle("Scaled Angular Deviation $x_p$ in Each (E_bin, θ_bin)", fontsize=16)

    fit_results = []

    for E_bin in range(len(E_bins_center)):
        for theta_bin in range(len(theta_bins_center)):
            print(f'Processing E_bin={E_bin}, theta_bin={theta_bin}')

            subset = df[(df['E_bin'] == E_bin) & (df['theta_bin'] == theta_bin)]
            if subset.empty:
                print(f"Empty bin for E_bin={E_bin}, θ_bin={theta_bin}")
                fit_results.append({
                    'E_bin': E_bin,
                    'theta_bin': theta_bin,
                    'A_c': np.nan, 'sigma_c': np.nan, 'gamma_c': np.nan,
                    'A_t': np.nan, 'sigma_t': np.nan, 'gamma_t': np.nan,
                    'chi2': np.nan, 'ndof': np.nan, 'chi2_ndof': np.nan,
                    'p_value': np.nan, 'n_entries': 0
                })
                continue

            n_bins = min(30, max(10, len(subset) // 5))
            xpMin, xpMax = subset["x_p"].min(), subset["x_p"].max()
            bins = np.logspace(np.log10(xpMin), np.log10(xpMax), n_bins)

            ax = axes[E_bin, theta_bin]
            ax.set_title(f"E_bin={E_bin}, θ_bin={theta_bin}", fontsize=12)
            ax.set_xlabel("$x_p$")
            ax.set_ylabel("$2\pi\, x_p \,P(x_p)$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 10)

            try:
                fit_result = fit_bin_two_stage(subset, bins)
                params = fit_result['params']
                chi2_red = fit_result['chi2_red']
                p_val = fit_result['p_value']

                print(f"Fit result for E_bin={E_bin}, θ_bin={theta_bin}: "
                    f"χ²/ndof = {fit_result['chi2']:.2f}/{len(fit_result['bin_centers']) - 6} = {chi2_red:.2f}")

                fit_results.append({
                    'E_bin': E_bin,
                    'theta_bin': theta_bin,
                    'A_c': params['A'] * params['F'],
                    'sigma_c': params['sigma_c'],
                    'gamma_c': params['gamma_c'],
                    'A_t': params['A'] * (1 - params['F']),
                    'sigma_t': params['sigma_t'],
                    'gamma_t': params['gamma_t'],
                    'chi2': fit_result['chi2'],
                    'ndof': len(fit_result['bin_centers']) - 6,
                    'chi2_ndof': chi2_red,
                    'p_value': p_val,
                    'n_entries': len(subset)
                })

                chi2_ndof_label = f"$\\chi^2$/ndof = {chi2_red:.2f}"

            except RuntimeError as e:
                print(f"Fit failed for E_bin={E_bin}, θ_bin={theta_bin}: {e}")
                fit_result = None
                chi2_ndof_label = "No fit (failed)"
                fit_results.append({
                    'E_bin': E_bin,
                    'theta_bin': theta_bin,
                    'A_c': np.nan, 'sigma_c': np.nan, 'gamma_c': np.nan,
                    'A_t': np.nan, 'sigma_t': np.nan, 'gamma_t': np.nan,
                    'chi2': np.nan, 'ndof': np.nan, 'chi2_ndof': np.nan,
                    'p_value': np.nan, 'n_entries': len(subset)
                })

            plot_fit_result(ax, subset, bins, fit_result, show_legend=False, chi2_ndof_label=chi2_ndof_label)
            fig_single, ax_single = plt.subplots(figsize=(5, 4))
            ax_single.set_title(f"E_bin={E_bin}, θ_bin={theta_bin}", fontsize=12)
            ax_single.set_xlabel("$x_p$")
            ax_single.set_ylabel("$2\pi\,\\times\,P(x_p)$")
            ax_single.set_xscale("log")
            ax_single.set_yscale("log")
            ax_single.set_ylim(1e-3, 10)

            plot_fit_result(ax_single, subset, bins, fit_result, show_legend=True, chi2_ndof_label=chi2_ndof_label)

            fig_single.tight_layout()
            subResPath = f"{resPath}bin_plots/"
            try : 
                if not os.path.exists(subResPath):
                    os.makedirs(subResPath)
            except FileExistsError as e:
                pass

            fig_single.savefig(f"{subResPath}E{E_bin}_theta{theta_bin}.png")
            fig_single.savefig(f"{subResPath}E{E_bin}_theta{theta_bin}.eps", format='eps')

            plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(resPath + "xp_HET_fits_allbins.png")
    figure_path = "/home/users/l/lebrun/figures/PSF/exp0.0/script/xp_HET_fits_allbins"
    fig.savefig(figure_path+'.eps', format='eps')
    fig.savefig(figure_path+'.png', format='png')
    plt.show()

    param_df = pd.DataFrame(fit_results)
    if paramPath:
        param_df.to_csv(paramPath, index=False)
        print(f"Fit parameters saved to {paramPath}")

    return param_df
            

def load_obs_time(target, beegfs_path="/dpnc/beegfs/dampe/tryf/ForKimberly/") :
    obsTime_path = beegfs_path + "obsTime/" + target + f"/obs_time_{target}.npz"
    obsTime = np.load(obsTime_path)
    
    hist_obsTime = obsTime['histogram']
    theta_bins_edge = obsTime['theta_edges']
    phi_bins_edge = obsTime['phi_edges']
    
    return hist_obsTime, theta_bins_edge, phi_bins_edge


if __name__ == "__main__":

    target = sys.argv[1] if len(sys.argv) > 1 else None

    dirDir= "/home/users/l/lebrun/"
    dataPath = "/dpnc/beegfs/dampe/tryf/ForKimberly/"
    exponent = 0
    figPath = dirDir + "figures/"
    figPath = figPath+f"PSF/exp{exponent:.1f}/script/"
    resPath = dirDir + f"results/PSF/exp{exponent:.1f}/script/"

    # Ensures fig and result paths exist
    try : 
        if not os.path.exists(resPath):
            os.makedirs(resPath)
    except FileExistsError as e:
        pass
    try : 
        if not os.path.exists(figPath):
            os.makedirs(figPath)
    except FileExistsError as e:
        pass

    dataPath_filtered = dataPath + "tt_filtered" + "_cos_theta_bins/"
    dataPath_filtered += f"exp{exponent:.1f}/"

    files_filtered = browse_files(dataPath_filtered + "*.root")
    print(f"Loading files: {files_filtered}")

    rdf_filtered = ROOT.RDataFrame("MC", files_filtered)

    # Extract only desired columns
    df = rdf_filtered.AsNumpy(["tt_ekin", "tt_theta", "tt_phi", "weight", "tt_isHET", "theta_BestSTK", "phi_BestSTK"])
    df = pd.DataFrame(df)

    nbins = 50

    # Preprocess and display infos
    df = preprocess_df(df)
    infos_df(df)

    df = compute_angular_error(df)

    # Plot angular error distribution
    figName = figPath + f"angular_error_distribution_exp{exponent:.1f}.png"
    plot_angular_error(df, figName)

    df, resolution_per_bin, resolution_per_bin_w = compute_resolution(df)
    fileName = resPath + f"resolution_per_bin.csv"
    fileName_w = resPath + f"resolution_per_bin_w.csv"

    print("Saving resolution matrices ...")
    save_df_to_csv(resolution_per_bin, fileName)
    save_df_to_csv(resolution_per_bin_w, fileName_w)

    print("Plotting resolutions...")
    figName = figPath+f"resolutionMap_HET_unweighted.png"
    figName_w = figPath+f"resolutionMap_HET_weighted.png"
    plot_resolution(resolution_per_bin, figName)
    print(f"Plotted resolution unweigthed to {figName}")
    plot_resolution(resolution_per_bin_w, figName_w)
    print(f"Plotted resolution weigthed to {figName_w}")

    # Compute scaled angular resolution
    df['x_p'] = df['angular_error_degrees'] / df["angular_resolution"]

    # Fit all bins
    paramPath = resPath + "fit_DK_params_all_bins.csv"

    if target is None :
        # If no target specified, fit PSF for all (E, cos(theta)) bins
        fits_df = fit_all_bins(df, resPath, paramPath)
    else :
        # If target specified, compute the exposure weighted PSF for all E bins

        path_effArea = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
        
        effArea_npz = np.load(path_effArea)
        effArea_table = effArea_npz['effArea'] # Shape (n_E_bins, n_costheta_bins, n_phi_bins) ~ (20, 10, 360)

        effArea_E_bins_edge = effArea_npz['E_bins_edge']
        effArea_E_bins_center = np.sqrt(effArea_E_bins_edge[:-1] * effArea_E_bins_edge[1:])  # GeV

        effArea_theta_bins_edge = effArea_npz['theta_bins_edge']
        effArea_phi_bins_edge = effArea_npz['phi_bins_edge']

        obsTime, obsT_theta_bins_edge, obsT_phi_bins_edge = load_obs_time(target, dataPath)
        
