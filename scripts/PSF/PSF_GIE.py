import ROOT, glob, os
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
theta_bins_center = (theta_bins_edge[:-1] + theta_bins_edge[1:])/2

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
    outliers = df[(df['angular_error_degrees'] > 10)]
    print("Outliers count:", outliers.shape[0])
    print("Outliers proportion : ", outliers.shape[0] / df.shape[0] *100, "%")

    df = df.loc[(df['angular_error_degrees'] < 10)]
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
    data, weights = np.array(data), np.array(weights)
    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    cumulative_weight = np.cumsum(weights_sorted)
    total_weight = cumulative_weight[-1]
    return np.interp(np.array(percentiles) / 100 * total_weight, cumulative_weight, data_sorted)

def weighted_angular_resolution_68(group):
    angles = group['angular_error_degrees'].values
    weights = group['weight'].values
    lower, upper = weighted_percentile(angles, weights, [16, 84])
    return (upper - lower) / 2

def angular_resolution_68(x):
    lower = np.percentile(x, 16)
    upper = np.percentile(x, 84)
    return (upper - lower) / 2

def compute_resolution(df): 
    # Assign bin indices
    df['E_bin'] = pd.cut(df['E_true'], bins=E_bins_edge, labels=False, include_lowest=True)
    df['theta_bin'] = pd.cut(df['cosTheta'], bins=theta_bins_edge, labels=False, include_lowest=True)

    # Drop rows that fall outside the bins (e.g., NaN bin index)
    df = df.dropna(subset=['E_bin', 'theta_bin']).copy() # Copy to avoid SettingWithCopyWarning

    # Make sure bin indices are integers
    df['E_bin'] = df['E_bin'].astype(int)
    df['theta_bin'] = df['theta_bin'].astype(int)

    # Group by (E_bin, theta_bin)
    grouped = df.groupby(['E_bin', 'theta_bin'])

    resolution_per_bin = grouped['angular_error_degrees'].apply(
        angular_resolution_68
    ).reset_index(name='angular_resolution')
    print(resolution_per_bin)

    resolution_per_bin_w = grouped.apply(
        weighted_angular_resolution_68
    ).reset_index(name='angular_resolution')
    print(resolution_per_bin_w)


    # Step 4: Merge back weighted version into original dataframe
    if 'angular_resolution' in df.columns:
        df = df.drop(columns='angular_resolution')
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


def king(xp, sigma, gamma):
    factor1 = 1/(2*np.pi*sigma*sigma)
    factor2 = (1 - 1/gamma)
    factor3 = (1 + 0.5*(1/gamma)*((xp/sigma)**2))**(-gamma)

    return factor1 * factor2 * factor3

def x_king(xp, sigma, gamma):
    return xp * king(xp, sigma, gamma)


def doubleKing(xp, A, F, sigma_c, gamma_c, sigma_t, gamma_t) :

    return A *(F * x_king(xp, sigma_c, gamma_c) + (1-F) * x_king(xp, sigma_t, gamma_t))

def chisq(data, model, error, dofs):
    residuals = np.abs(data - model)
    chi_squared = np.sum((residuals)**2 / (error**2))
    p_value = 1 - chi2.cdf(chi_squared, dofs)
    return chi_squared, p_value

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


def fit_xp_bin(subset, bins) :
    
    bin_centers, counts, errors, counts_norm, errors_norm = get_histogram_normed(subset, bins)

     
    epsilon = 1e-5
    valid = (counts > 5)

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

def plot_fit_result(ax, bin_centers, counts_norm, errors_norm, popt=None, show_legend=True, chi2_ndof_label=None):
    """
    Plot histogram data and fit result on a given axis.
    """
    ax.errorbar(bin_centers, 2*np.pi*counts_norm, yerr=2*np.pi*errors_norm, fmt='o', color='royalblue', alpha=0.7, label="Data", markersize=3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-3, 10)
    ax.set_xlabel("$x_p$")
    ax.set_ylabel("$2\pi\,\\times\,P(x_p)$")

    if popt is not None:
        x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
        y_fit = doubleKing(x_fit, *popt)

        A, F, sigma_c, gamma_c, sigma_t, gamma_t = popt
        y_fit1 = A * (F * x_king(x_fit, sigma_c, gamma_c))
        y_fit2 = A * (1 - F) * x_king(x_fit, sigma_t, gamma_t)

        label_fit = 'Fit'
        if chi2_ndof_label:
            label_fit += f' ({chi2_ndof_label})'

        ax.plot(x_fit, 2*np.pi*y_fit, color='darkred', lw=2, label=label_fit)
        ax.plot(x_fit, 2*np.pi*y_fit1, color='green', lw=2, label='King core')
        ax.plot(x_fit, 2*np.pi*y_fit2, color='orange', lw=2, label='King tail')

        if show_legend:
            ax.legend()

def fit_all_bins(df, resPath, paramPath=None):
    """
    Fit the double King distribution to binned data in (E_bin, θ_bin) space and plot the results.

    This function iterates over all combinations of energy and angle bins in the input DataFrame,
    fits the `doubleKing` function to the distribution of `x_p` within each bin, plots the fits,
    and optionally saves the fit parameters to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the columns: 'E_bin', 'theta_bin', and 'x_p'.
        Each row represents one measurement/event with precomputed bin assignments.
    
    figName : str
        File path to save the full figure showing all subplots of the fits.

    paramPath : str or None, optional
        If provided, the fit parameters for each bin are saved to a CSV file at this location.
        Default is None (no file saved).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the fit results for each (E_bin, θ_bin) combination, with columns:
        ['E_bin', 'theta_bin', 'A', 'F', 'sigma_c', 'gamma_c', 'sigma_t', 'gamma_t',
         'chi2', 'ndof', 'chi2_ndof', 'p_value', 'n_entries']

        If a fit fails (e.g., due to an empty bin), parameter fields are filled with NaN.

    Notes
    -----
    - The function uses logarithmic binning for the `x_p` variable in each subset.
    - Fits are performed using `fit_xp_bin`, which must return the optimal parameters (`popt`),
      chi-squared, p-value, and normalized histogram data.
    - The `EmptyBinException` is handled gracefully to allow processing to continue.
    - The plotting uses log-log scale with consistent axis limits across subplots.

    Examples
    --------
    >>> param_df = fit_all_bins(data, "fit_results.png", "fit_parameters.csv")
    >>> print(param_df.head())
    """

    fig, axes = plt.subplots(n_E_bins, n_theta_bins, figsize=(n_E_bins * 4, n_theta_bins * 3), sharex=True, sharey=True)
    fig.suptitle("Scaled Angular Deviation x_p in Each (E_bin, θ_bin)", fontsize=16)

    fit_results = []

    for E_bin in range(len(E_bins_center)):
        for theta_bin in range(len(theta_bins_center)):
            print(f'Processing E_bin={E_bin}, theta_bin={theta_bin}')

            subset = df[(df['E_bin'] == E_bin) & (df['theta_bin'] == theta_bin)] 
            n_bins = min(30, max(10, len(subset) // 5))
            print(f"Number of bins generated : {n_bins}")

            xpMin = subset["x_p"].min()
            xpMax = subset["x_p"].max()
            bins = np.logspace(np.log10(xpMin), np.log10(xpMax), n_bins)

            ax = axes[E_bin, theta_bin]
            ax.set_title(f"E_bin={E_bin}, θ_bin={theta_bin}", fontsize=16)
            ax.set_xlabel("$x_p$")
            ax.set_ylabel("$D(x_p)$")
            ax.set_yscale('log')
            ax.set_xscale("log")
            ax.set_ylim(1e-3, 10)

            try:
                popt, chi_squared, p_val, bin_centers, counts_norm, errors_norm = fit_xp_bin(subset, bins)
                ndof = len(bin_centers) - len(popt)
                chi_squared_str = f"$\chi^2$/ndof = {chi_squared:.2f}/{ndof} = {chi_squared/ndof:.2f}"

                # Save parameters and metrics
                fit_results.append({
                    'E_bin': E_bin,
                    'theta_bin': theta_bin,
                    'A': popt[0],
                    'F': popt[1],
                    'sigma_c': popt[2],
                    'gamma_c': popt[3],
                    'sigma_t': popt[4],
                    'gamma_t': popt[5],
                    'chi2': chi_squared,
                    'ndof': ndof,
                    'chi2_ndof': chi_squared / ndof,
                    'p_value': p_val,
                    'n_entries': len(subset)
                })

            except EmptyBinException as e:
                print(e)
                popt = None
                bin_centers, counts, errors, counts_norm, errors_norm = get_histogram_normed(subset, bins)

                # Log failed bin
                fit_results.append({
                    'E_bin': E_bin,
                    'theta_bin': theta_bin,
                    'A': np.nan, 'F': np.nan, 'sigma_c': np.nan,
                    'gamma_c': np.nan, 'sigma_t': np.nan, 'gamma_t': np.nan,
                    'chi2': np.nan, 'ndof': np.nan, 'chi2_ndof': np.nan,
                    'p_value': np.nan,
                    'n_entries': len(subset)
                })
            
            chi2_ndof_label = None
            empty = False
            if popt is not None and ndof > 0:
                chi2_ndof_label = f"$\\chi^2$/ndof = {chi_squared/ndof:.1f}"
                print(f"Fit result for E_bin={E_bin}, θ_bin={theta_bin}: χ2/ndof = {chi_squared:.2f}/{ndof} = {chi_squared/ndof:.2f}")
                if chi_squared/ndof < 0.5 :
                    print("WARNING : χ2/ndof < 0.5, fit may be overfitting")
                elif chi_squared/ndof > 2 :
                    print("WARNING : χ2/ndof > 2, fit may be underfitting")
            else:
                chi2_ndof_label = "No fit (empty bin)"
                empty = True
                print(f"Empty bin for E_bin={E_bin}, θ_bin={theta_bin}")

            # Plot on big grid
            plot_fit_result(ax, bin_centers, counts_norm, errors_norm, popt=popt, show_legend=False, chi2_ndof_label=chi2_ndof_label)

            # Save individual plot
            if not empty :
                fig_single, ax_single = plt.subplots(figsize=(5, 4))
                ax_single.set_title(f"E_bin={E_bin}, θ_bin={theta_bin}")


                plot_fit_result(ax_single, bin_centers, counts_norm, errors_norm, popt=popt, show_legend=True, chi2_ndof_label=chi2_ndof_label)

                fig_single.tight_layout()
                subResPath = f"{resPath}bin_plots/"
                try : 
                    if not os.path.exists(subResPath):
                        os.makedirs(subResPath)
                except FileExistsError as e:
                    pass

                fig_single.savefig(f"{subResPath}E{E_bin}_theta{theta_bin}.png")

                plt.close(fig_single)

            # if popt is not None:
            #     x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
            #     y_fit = doubleKing(x_fit, *popt)

            #     ax.plot(x_fit, y_fit, color='darkred', lw=2, label=f'Fit\n{chi_squared_str}')

            #     A, F, sigma_c, gamma_c, sigma_t, gamma_t = popt
            #     y_fit1 = A * (F * x_king(x_fit, sigma_c, gamma_c))
            #     y_fit2 = A * (1 - F) * x_king(x_fit, sigma_t, gamma_t)

            #     ax.plot(x_fit, y_fit1, color='green', lw=2, label='King core')
            #     ax.plot(x_fit, y_fit2, color='orange', lw=2, label='King tail')
            #     ax.legend()

    fig.tight_layout()
    fig.savefig(resPath + "xp_HET_fits_allbins.png")
    plt.show()

    # Save fit parameters to file
    param_df = pd.DataFrame(fit_results)
    if paramPath:
        param_df.to_csv(paramPath, index=False)
        print(f"Fit parameters saved to {paramPath}")

    return param_df
            



if __name__ == "__main__":

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

    fits_df = fit_all_bins(df, resPath, paramPath)
