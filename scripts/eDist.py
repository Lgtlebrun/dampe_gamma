import ROOT, glob, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

nbins = 50

def preprocess_df(df:pd.DataFrame) :

    df = df.rename(columns={
        'tt_theta': 'theta_True',
        'tt_ekin': 'E_true',
        'tt_bgoTotalEcorr_GeV': 'E_reco',
        'weight': 'weight'
    })

    df['E_true'] = df['E_true'] * 1e-3

    df = df[df['E_true'] <= 1e4]
    df['E_reco'] = df['tt_bgoTotalEcorr_GeV']

    df['theta_True_deg']= np.rad2deg(df['theta_True'])
    df['cosTheta'] = np.cos(df['theta_True'])

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

    if 'E_reco' in df.columns:
        high_energy_mask = df['E_reco'] <= 1e4  # 10 TeV
        print(f"Number of events with E_reco <= 10 TeV: {high_energy_mask.sum()}")
        
        # Common energy ranges
        gev_ranges = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
        for low, high in gev_ranges:
            mask = (df['E_reco'] >= low) & (df['E_reco'] < high)
            count = mask.sum()
            print(f"  Events with E_reco in [{low}-{high}) GeV: {count}")
    
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

# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
# Load the effective area table
EFF_AREA_NPZ = np.load(PATH_EFFAREA)

# Get bins edges and centers
E_bins_edge = EFF_AREA_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(E_bins_edge[:-1] * E_bins_edge[1:])  # GeV

cos_theta_min = 0.5
cos_theta_max = 1
n_theta_bins = 5
theta_bins_edge = np.linspace(cos_theta_min, cos_theta_max, n_theta_bins + 1)
theta_bins_center = (theta_bins_edge[:-1] + theta_bins_edge[1:])/2


n_Ediff_bins = 100
Ediff_min = df['Ediff'].min()
Ediff_max = df['Ediff'].max()
Ediff_bins = np.linspace(Ediff_min, Ediff_max, n_Ediff_bins + 1)

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



# %%
def symmetric_containment_radius(data, weights=None, containment=0.68):
    """Find symmetric radius around 0 containing `containment` fraction of data."""

    data = np.array(data)

    if weights is None:
        weights = np.ones(data.shape)

    weights = np.array(weights)
    if len(data) == 0 or np.all(weights == 0):
        return np.nan
    
    # Sort by absolute distance from 0
    abs_sorted = np.abs(data)
    sorter = np.argsort(abs_sorted)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    
    cumsum = np.cumsum(weights_sorted)
    total_weight = cumsum[-1]
    
    # Find the smallest symmetric radius around 0 containing the desired weight
    idx = np.searchsorted(cumsum, containment * total_weight)
    
    if idx >= len(data_sorted):
        return np.nan
    
    # The corresponding symmetric radius around 0
    return np.abs(data_sorted[idx])


# %%
# Plot Ediff for each E, cosTheta bin ()
fig, axes = plt.subplots(n_E_bins, n_theta_bins, figsize=(12, 10), sharex=True, sharey=True)
for (E_bin, theta_bin), group in grouped:
    ax = axes[E_bin, theta_bin]
    ax.hist(group['Ediff'], bins=Ediff_bins, density=True, alpha=0.7)
    ax.set_title(f"E: {E_bins_center[E_bin]:.1f} GeV, cos(θ): {theta_bins_center[theta_bin]:.2f}")
    ax.set_xlabel("E_reco - E_true / E_true")
    ax.set_ylabel("Density")
    ax.grid(True)
plt.tight_layout()
plt.savefig(figPath + "Ediff_per_E_cosTheta.png")
plt.show()

# Plot them one by one
for E_bin in range(n_E_bins):
    for theta_bin in range(n_theta_bins):
        group = grouped.get_group((E_bin, theta_bin))
        if group.empty:
            continue
        
        plt.figure(figsize=(8, 6))
        plt.hist(group['Ediff'], bins=Ediff_bins, density=True, alpha=0.7)
        # Draw 16 and 84 percentiles
        p16 = np.percentile(group['Ediff'], 16)
        p84 = np.percentile(group['Ediff'], 84)
        plt.axvline(p16, color='red', linestyle='--', label='16th percentile')
        plt.axvline(p84, color='blue', linestyle='--', label='84th percentile')
        # Draw horizontal arrow between 16 and 84 percentiles
        plt.arrow(p16, 0.1, p84 - p16, 0, head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
        plt.axvline(0, color='black', linestyle='-', label='E_reco = E_true')
        # Draw 68% containment radius
        r = symmetric_containment_radius(group['Ediff'], group['weight'], containment=0.68)
        plt.annotate(
                        '', xy=(r, 1), xytext=(-r, 1),
                        arrowprops=dict(arrowstyle='<->', color='purple', lw=2, label="68% radius"),
                    )
        
        plt.legend()
        plt.title(f"E: {E_bins_center[E_bin]:.1f} GeV, cos(θ): {theta_bins_center[theta_bin]:.2f}")
        plt.xlabel("E_reco - E_true / E_true")
        plt.ylabel("Density")
        plt.grid(True)
        plt.savefig(figPath + f"Ediff_E{E_bins_center[E_bin]:.1f}_cosTheta{theta_bins_center[theta_bin]:.2f}.png")
        plt.show()


# %%
# TODO : Remove ? 
def weighted_percentile(data, weights, percentiles):
    """Compute weighted percentiles of data."""
    data, weights = np.array(data), np.array(weights)
    if len(data) == 0:
        return [np.nan for _ in percentiles]

    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    
    cumsum = np.cumsum(weights_sorted)
    cumsum /= cumsum[-1]  # normalize to 1

    return np.interp(percentiles, cumsum, data_sorted)

def weighted_HBW(g):
    w = g['weight']
    if len(g) < 2 or np.all(w == 0) or np.isnan(w).all():
        return np.nan
    p16, p84 = weighted_percentile(g['Ediff'], w, [0.16, 0.84])
    return 0.5 * (p84 - p16)

#Apply resolution calculation : HBW of 68% containment interval 
#New version : cvompute symmetric radius
resolution_per_bin = grouped['Ediff'].apply(
    lambda x: 0.5 * (np.percentile(x, 84) - np.percentile(x, 16))
).reset_index(name='resolution')

resolution_per_bin_w = grouped.apply(weighted_HBW).reset_index(name='resolution_w')

resolution_per_bin_r = grouped.apply(
    lambda x: symmetric_containment_radius(x['Ediff'], containment=0.68)
).reset_index(name='resolution')

resolution_per_bin_w_r = grouped.apply(
    lambda x: symmetric_containment_radius(x['Ediff'], weights=x['weight'] ,containment=0.68)
).reset_index(name='resolution_w')

results_df = pd.merge(resolution_per_bin, resolution_per_bin_w, on=['E_bin', 'theta_bin'], how='left')


# %%
# Build the resolution grid directly
resolution_grid = np.full((n_E_bins, n_theta_bins), np.nan)
for _, row in resolution_per_bin.iterrows():
    resolution_grid[int(row['E_bin']), int(row['theta_bin'])] = row['resolution']

resolution_grid_w = np.full((n_E_bins, n_theta_bins), np.nan)
for _, row in resolution_per_bin_w.iterrows():
    resolution_grid_w[int(row['E_bin']), int(row['theta_bin'])] = row['resolution_w']

resolution_grid_r = np.full((n_E_bins, n_theta_bins), np.nan)
for _, row in resolution_per_bin_r.iterrows():
    resolution_grid_r[int(row['E_bin']), int(row['theta_bin'])] = row['resolution']

resolution_grid_w_r = np.full((n_E_bins, n_theta_bins), np.nan)
for _, row in resolution_per_bin_w_r.iterrows():
    resolution_grid_w_r[int(row['E_bin']), int(row['theta_bin'])] = row['resolution_w']

# Prepare edges for pcolormesh
E_edges = E_bins_edge

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.pcolormesh(E_edges, theta_bins_edge, resolution_grid.T, shading='auto', cmap='viridis')

# Log scale for energy
ax.set_xscale('log')
ax.set_xlabel("True Energy [GeV]")
ax.set_ylabel("cos(θ)")
ax.set_title("HET Energy Resolution Map")

# Colorbar
cb = fig.colorbar(c, ax=ax)
cb.set_label("Energy Resolution")

plt.tight_layout()
# Create dir if does not exist
try:
    os.makedirs(figPath)
except FileExistsError:
    pass
fig.savefig(figPath + "HET_Energy_Resolution_Map.png", dpi=400)
plt.show()


# %%
### Plot resolution_grid and resolution_grid_w on the same figure

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# Plot resolution_grid
c1 = ax1.pcolormesh(E_edges, theta_bins_edge, resolution_grid.T, shading='auto', cmap='viridis')
# Log scale for energy
ax1.set_xscale('log')
ax1.set_xlabel("True Energy [GeV]")
ax1.set_ylabel("cos(θ)")
ax1.set_title("HET Energy Resolution Map (Unweighted, Percentiles)")
# Colorbar
cb1 = fig.colorbar(c1, ax=ax1)
cb1.set_label("Energy Resolution")
# Plot resolution_grid_w
c2 = ax2.pcolormesh(E_edges, theta_bins_edge, resolution_grid_w.T, shading='auto', cmap='viridis')
# Log scale for energy
ax2.set_xscale('log')
ax2.set_xlabel("True Energy [GeV]")
ax2.set_ylabel("cos(θ)")
ax2.set_title("HET Energy Resolution Map (Weighted, Percentiles)")
# Colorbar
cb2 = fig.colorbar(c2, ax=ax2)
cb2.set_label("Energy Resolution")
plt.tight_layout()
fig.savefig(figPath + "HET_Energy_Resolution_Map_comparison.png", dpi=400)
plt.show()

# %%
### Plot resolution_grid and resolution_grid_w on the same figure

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# Plot resolution_grid
c1 = ax1.pcolormesh(E_edges, theta_bins_edge, resolution_grid_r.T, shading='auto', cmap='viridis')
# Log scale for energy
ax1.set_xscale('log')
ax1.set_xlabel("True Energy [GeV]")
ax1.set_ylabel("cos(θ)")
ax1.set_title("HET Energy Resolution Map (Unweighted)")
# Colorbar
cb1 = fig.colorbar(c1, ax=ax1)
cb1.set_label("Energy Resolution")
# Plot resolution_grid_w
c2 = ax2.pcolormesh(E_edges, theta_bins_edge, resolution_grid_w_r.T, shading='auto', cmap='viridis')
# Log scale for energy
ax2.set_xscale('log')
ax2.set_xlabel("True Energy [GeV]")
ax2.set_ylabel("cos(θ)")
ax2.set_title("HET Energy Resolution Map (Weighted)")
# Colorbar
cb2 = fig.colorbar(c2, ax=ax2)
cb2.set_label("Energy Resolution")
plt.tight_layout()
fig.savefig(figPath + "HET_Energy_Resolution_Map_comparison_r.png", dpi=400)
plt.show()

# %% [markdown]
# ### CONCLUSION
# - Better resolution (lower values) when HET + LET considered
# - Expected a minimum at higher energy ? Not the case

# %% [markdown]
# ## Save map for future use 

# %%
np.savez_compressed(
    resultPath + "energy_resolution_grid_LET.npz",
    resolution_grid=resolution_grid,
    E_bins_edge=E_bins_edge,
    theta_bins_edge=theta_bins_edge
)

# %% [markdown]
# ## Compute scaled energy deviation

# %%
df['x_D'] = df['Ediff'] / df['resolution']

# %% [markdown]
# ## Plot x_D

# %%
# Plot x_D for all 20 E, theta bins
# Determine subplot grid size
n_rows = n_E_bins
n_cols = n_theta_bins

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
fig.suptitle("Scaled Energy Deviation x_D in Each (E_bin, θ_bin)", fontsize=16)

# Plot each bin
for i in range(n_E_bins):
    for j in range(n_theta_bins):
        ax = axes[i, j]
        subset = df[(df['E_bin'] == i) & (df['theta_bin'] == j)]

        if not subset.empty:
            ax.hist(subset['x_D'], bins=50, color='royalblue', alpha=0.7, density=True)
            ax.set_title(f"E_bin={i}, θ_bin={j}", fontsize=9)
        else:
            ax.set_visible(False)

        if i == n_rows - 1:
            ax.set_xlabel("x_D")
        if j == 0:
            ax.set_ylabel("$D(x)$")
        ax.set_yscale('log')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [markdown]
# ## Fit with Fermi's functions

# %%
from scipy.special import gamma
from scipy.stats import chi2

def asymetric_gaussian(x, sigma, k, b, p):
    factorK = k / (1 + k**2)
    factorP = p / (sigma * gamma(1 / p))
    gaus = np.where(
        x - b >= 0,
        np.exp(-(k * np.abs(x - b) / sigma)**p),
        np.exp(-((np.abs(x - b) / (sigma * k))**p))
    )
    return factorK * factorP * gaus

# Define the full fit model
def fitDensity(x, F, sigma1, k1, b1, p1, sigma2, k2, b2, p2):
    return (
        F * asymetric_gaussian(x, sigma1, k1, b1, p1) +
        (1 - F) * asymetric_gaussian(x, sigma2, k2, b2, p2)
    )

def chisq(data, model, error, dofs):
    residuals = np.abs(data - model)
    chi_squared = np.sum((residuals)**2 / (error**2))
    p_value = 1 - chi2.cdf(chi_squared, dofs)
    return chi_squared, p_value


# %%
from scipy.optimize import curve_fit
# Select a specific bin (E_bin and theta_bin)
E_bin = 0
theta_bin = 1
n_bins = 100
subset = df[(df['E_bin'] == E_bin) & (df['theta_bin'] == theta_bin) & (df['x_D'] > -15)] 
counts, bin_edges = np.histogram(subset['x_D'], bins=n_bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_widths = bin_edges[1:] - bin_edges[:-1]
errors = np.sqrt(counts)

# Normalize counts and errors
total = np.sum(counts)
counts_norm = counts / (bin_widths * total) # TODO: check if this is correct
errors_norm = errors / (bin_widths * total)

# ---- Initial guesses ----
F_init = 0.5
mean = np.average(subset['x_D'])
std = np.std(subset['x_D'])

print(f"Numbers of bins under 5 counts threshold : {len(counts[counts < 5])}")

threshold = 5e-3 * np.max(counts)

# Mask: counts > 5 and density > 10⁻3
mask = (counts > threshold) & (counts_norm > 0.001) & (errors_norm < 0.5*counts_norm)
bin_centers_masked = bin_centers[mask]
print("Number of bins remaining after mask :", len(bin_centers_masked))
counts_norm_masked = counts_norm[mask]
errors_norm_masked = errors_norm[mask]

# Updated initial guesses
mean = np.average(subset['x_D'])
std = np.std(subset['x_D'])
p0 = [
    0.5,        # F
    std / 2, 1.0, mean - std/2, 2.0,  # sigma1, k1, b1, p1
    std,     1.0, mean + std/2, 2.0   # sigma2, k2, b2, p2
]

# Bounds: to constrain parameters to physical values
bounds = (
    [0,   1e-3, 0.1, mean - 3*std, 0.5,  1e-3, 0.1, mean - std, 0.5],   # Lower bounds
    [1,   10.0, 10.0, mean,       10.0, 10.0, 10.0, mean + 3*std, 10.0]  # Upper bounds
)

# Fit
try:
    popt, pcov = curve_fit(
        fitDensity,
        bin_centers_masked,
        counts_norm_masked,
        p0=p0,
        sigma=errors_norm_masked,
        bounds=bounds,
        absolute_sigma=True,
        maxfev=20000
    )
except RuntimeError as e:
    print("Fit failed:", e)
    popt = None

# Compute expected values from fit
fit_vals = fitDensity(bin_centers_masked, *popt)
fit_vals_counts = fit_vals * (bin_widths[mask] * total)  # Convert back to counts

F, sigma1, k1, b1, p1, sigma2, k2, b2, p2 = popt
print("F:", F, "bounds:", bounds[0][0], bounds[1][0])
print("sigma1:", sigma1, "bounds:", bounds[0][1], bounds[1][1])
print("k1:", k1, "bounds:", bounds[0][2], bounds[1][2])
print("b1:", b1, "bounds:", bounds[0][3], bounds[1][3])
print("p1:", p1, "bounds:", bounds[0][4], bounds[1][4])
print("sigma2:", sigma2, "bounds:", bounds[0][5], bounds[1][5])
print("k2:", k2, "bounds:", bounds[0][6], bounds[1][6])
print("b2:", b2, "bounds:", bounds[0][7], bounds[1][7])
print("p2:", p2, "bounds:", bounds[0][8], bounds[1][8])


# Compute chi-squared
ndof = len(bin_centers_masked) - len(popt)  # number of degrees of freedom
chi_squared, p_val = chisq(counts[mask], fit_vals_counts, errors[mask], ndof)
chi_squared_str = f"$\chi^2$/ndof = {chi_squared:.2f}/{ndof} = {chi_squared/ndof:.2f}"

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(bin_centers_masked, counts_norm_masked, yerr=errors_norm_masked, fmt='o', color='royalblue', alpha=0.7, label="Data", markersize=3)

if popt is not None:
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_fit = fitDensity(x_fit, *popt)
    ax.plot(x_fit, y_fit, color='darkred', lw=2, label=f'Fit\n{chi_squared_str}')
    F, sigma1, k1, b1, p1, sigma2, k2, b2, p2 = popt
    y_fit1 = F * asymetric_gaussian(x_fit, sigma1, k1, b1, p1)
    y_fit2 = (1 - F) * asymetric_gaussian(x_fit, sigma2, k2, b2, p2)
    # Create masks for each Gaussian component based on the condition
    mask1 = y_fit1 > np.min(y_fit)  # Gaussian 1 condition: x - b1 >= 0
    mask2 = y_fit2 > np.min(y_fit)   # Gaussian 2 condition: x - b2 < 0

    # Apply the masks to plot only the regions where the conditions are satisfied
    ax.plot(x_fit[mask1], y_fit1[mask1], color='green', lw=2, label='Gaussian 1')
    ax.plot(x_fit[mask2], y_fit2[mask2], color='orange', lw=2, label='Gaussian 2')

ax.set_title(f"$x_D$ Distribution with Fit (E_bin={E_bin}, θ_bin={theta_bin})", fontsize=16)
ax.set_xlabel("$x_D$")
ax.set_ylabel("$D(x_D)$")
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Try with Chinese people function

# %%
def R(x, x0, sigma, gamma):
    return np.exp(-0.5 * np.abs((x - x0) / sigma)**gamma)

def D(x, x0, xbarL, xbarR, sigmaL, gammaL, sigmaI, gammaI, sigmaR, gammaR, N):
    # Piecewise evaluation
    D_vals = np.zeros_like(x)
    mask_L = x - x0 < xbarL
    mask_I = (x - x0 >= xbarL) & (x - x0 <= xbarR)
    mask_R = x - x0 > xbarR

    NL = N * R(xbarL, x0, sigmaI, gammaI) / R(xbarL, x0, sigmaL, gammaL)
    NR = N * R(xbarR, x0, sigmaI, gammaI) / R(xbarR, x0, sigmaR, gammaR)

    D_vals[mask_L] = NL * R(x[mask_L], x0, sigmaL, gammaL)
    D_vals[mask_I] = N * R(x[mask_I], x0, sigmaI, gammaI)
    D_vals[mask_R] = NR * R(x[mask_R], x0, sigmaR, gammaR)

    return D_vals

# %% [markdown]
# ### First step : fit with all degrees of freedom (first guess)

# %%
def curve_fit_custom(*args, tol=0.05, **kwargs):
    # Run curve fitting
    popt, pcov = curve_fit(*args, **kwargs)

    # Warn if near bounds
    if "bounds" in kwargs:
        lower_bounds, upper_bounds = kwargs["bounds"]
        for i, par in enumerate(popt):
            lb, ub = lower_bounds[i], upper_bounds[i]
            if np.isfinite(lb) and lb != 0:
                if np.abs(par - lb) < tol * np.abs(lb):
                    print(f"WARNING: Parameter {i} = {par:.4g} is within {tol*100:.1f}% of its lower bound ({lb})")
            if np.isfinite(ub) and ub != 0:
                if np.abs(par - ub) < tol * np.abs(ub):
                    print(f"WARNING: Parameter {i} = {par:.4g} is within {tol*100:.1f}% of its upper bound ({ub})")
    
    return popt, pcov

def random_walk_fit(model, x, y, yerr, p0, bounds, norm_array, n_trials=30, noise_scale=0.1, verbose=False):
    best_popt = None
    best_chi2 = np.inf
    best_fit_vals = None

    lower_bounds, upper_bounds = bounds

    for trial in range(n_trials):
        p0_perturbed = []
        for i, p in enumerate(p0):
            if p != 0:
                perturbed = p * (1 + noise_scale * np.random.uniform(-1, 1))
            else:
                perturbed = noise_scale * np.random.randn()

            # Clip to stay within bounds
            perturbed = np.clip(perturbed, lower_bounds[i], upper_bounds[i])
            p0_perturbed.append(perturbed)

        try:
            popt, pcov = curve_fit(
                model,
                x,
                y,
                p0=p0_perturbed,
                sigma=yerr,
                bounds=bounds,
                absolute_sigma=True,
                maxfev=30000
            )
            fit_vals = model(x, *popt)
            fit_vals_counts = fit_vals * norm_array  # Convert back to counts
            counts = y * norm_array  # Convert back to counts
            yerr_counts = yerr * norm_array  # Convert back to counts
            chi2_val, p_val = chisq(counts, fit_vals_counts, yerr_counts, len(x) - len(popt))
            #np.sum((y - fit_vals) ** 2 / (yerr ** 2))

            if chi2_val < best_chi2:
                best_chi2 = chi2_val
                best_popt = popt
                best_fit_vals = fit_vals

            if verbose:
                print(f"Trial {trial}: χ² = {chi2_val:.2f}")

        except RuntimeError:
            if verbose:
                print(f"Trial {trial}: Fit failed.")
            continue

    return best_popt, best_fit_vals, best_chi2

def random_start_random_walk_fit(
    model, x, y, yerr, bounds, norm_array, n_trials=30, noise_scale=0.1, verbose=False
):
    # Generate random starting points within bounds
    lower_bounds, upper_bounds = bounds
    p0_randoms = []
    # Generate n_trials random starting points
    for _ in range(n_trials):
        p0_random = []
        for i in range(len(lower_bounds)):
            if np.isfinite(lower_bounds[i]) and np.isfinite(upper_bounds[i]):
                p0_random.append(np.random.uniform(lower_bounds[i], upper_bounds[i]))
            else:
                p0_random.append(np.random.randn())  # Random value if bounds are not finite
        p0_randoms.append(p0_random)
    best_popt = None
    best_chi2 = np.inf
    best_fit_vals = None
    for p0 in p0_randoms:
        try:
            _popt, _fit_vals, _chi2 = random_walk_fit(
                model,
                x,
                y,
                yerr,
                p0,
                bounds,
                norm_array,
                n_trials=30,
                noise_scale=noise_scale,
                verbose=verbose
            )

            if _chi2 < best_chi2:
                best_chi2 = _chi2
                best_popt = _popt
                best_fit_vals = _fit_vals
                if verbose:
                    print(f"New best parameters: {best_popt}")
                    print(f"New best fit values: {best_fit_vals}")
                    print(f"New best χ²: {_chi2:.2f}")
                else: 
                    print(f"New best value : χ² = {_chi2:.2f}")

        except RuntimeError:
            if verbose:
                print("Fit failed.")
            continue
        except ValueError:
            if verbose:
                print("ValueError in fit.")
            continue
    

    return best_popt, best_fit_vals, best_chi2


# %%


# --- Initial Guesses ---
x0_guess = bin_centers[np.argmax(counts_norm)]
xbarL_guess = -10.01  # reasonable empirical cutoff
xbarR_guess = -0.2  # reasonable empirical cutoff
# p0 = [
#     x0_guess, xbarL_guess, xbarR_guess,  # x0, xbarL
#     0.4, 1.5,   # sigmaL, gammaL
#     0.4, 2.0,   # sigmaI, gammaI
#     0.4, 1.5,   # sigmaR, gammaR
#     1.0, 1.0, 1.0  # NL, NI, NR
# ]

p0 = [
    x0_guess, xbarL_guess, xbarR_guess,  # x0, xbarL
    7.54, 1.86,   # sigmaL, gammaL
    0.4, 2.0,   # sigmaI, gammaI
    0.4, 1.5,   # sigmaR, gammaR
    1.0  # N
]

# --- Bounds (to ensure positive widths and normalizations) ---
#   x0, xbarL, xbarR, sigmaL, gammaL, sigmaI, gammaI, sigmaR, gammaR, NL, NI, NR
bounds = (
    [-20, -15, -3,  0.001, 0.01,  1e-8, 0.01,  0.01, 0.4,  0],
    [ 20, 0, 1,  13,  10,  6, 10, 6, 10, 100]
)

norm_array = bin_widths[mask] * total

# Fit
try:
    popt, fit_vals, chi_squared = random_start_random_walk_fit(
    D,
    bin_centers_masked,
    counts_norm_masked,
    errors_norm_masked,
    bounds,
    norm_array,
    noise_scale=0.2,  # Increase if still stuck
    verbose=False,
    n_trials=100,  # Increase if still stuck
)

except RuntimeError as e:
    print("Fit failed:", e)
    popt = None

print(popt)

x0, xbarL, xbarR, sigmaL, gammaL, sigmaI, gammaI, sigmaR, gammaR, N = popt
NL = N * R(xbarL, x0, sigmaI, gammaI) / R(xbarL, x0, sigmaL, gammaL)
NR = N * R(xbarR, x0, sigmaI, gammaI) / R(xbarR, x0, sigmaR, gammaR)

print(f"Fit parameters:")
print(f"x0: {x0}, xbarL: {xbarL}, xbarR: {xbarR}")
print(f"sigmaL: {sigmaL}, gammaL: {gammaL}")
print(f"sigmaI: {sigmaI}, gammaI: {gammaI}")
print(f"sigmaR: {sigmaR}, gammaR: {gammaR}")
print(f"NL: {NL}, NI: {N}, NR: {NR}")


# Compute chi-squared

ndof = len(bin_centers_masked) - len(popt)  # number of degrees of freedom
p_value = 1 - chi2.cdf(chi_squared, ndof)

chi_squared_str = f"$\chi^2$/ndof = {chi_squared:.2f}/{ndof} = {chi_squared/ndof:.2f}"
print(f"P-value : ", p_value)
if p_value < 1e-16:
    prob_str = "p-value < 1e-16"
else:
    prob_str = f"p-value = {p_value:.2g}"

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(bin_centers_masked, counts_norm_masked, yerr=errors_norm_masked, fmt='o', color='royalblue', alpha=0.7, label="Data", markersize=3)

if popt is not None:
    x_fit = np.linspace(np.min(bin_centers_masked), np.max(bin_centers_masked), 1000)
    y_fit = D(x_fit, *popt)
    ax.plot(x_fit, y_fit, color='darkred', lw=2, label=f'Fit\n{chi_squared_str}\n{prob_str}')
    # Optional: show components
    mask_L = x_fit - x0 < xbarL
    mask_I = (x_fit - x0 >= xbarL) & (x_fit - x0 <= xbarR)
    mask_R = x_fit - x0 > xbarR
    ax.plot(x_fit[mask_L], NL * R(x_fit[mask_L], x0, sigmaL, gammaL), color='green', label='Left')
    ax.plot(x_fit[mask_I], N * R(x_fit[mask_I], x0, sigmaI, gammaI), color='blue', label='Core')
    ax.plot(x_fit[mask_R], NR * R(x_fit[mask_R], x0, sigmaR, gammaR), color='orange', label='Right')


ax.set_title(f"$x_D$ Distribution with Fit (E_bin={E_bin}, θ_bin={theta_bin})", fontsize=16)
ax.set_xlabel("$x_D$")
ax.set_ylabel("$D(x_D)$")
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
fig.savefig(figPath + f"x_D_HET_fitE{E_bin}theta{theta_bin}.png", dpi=400)
plt.show()

# %% [markdown]
# ### Step 2 : Plot all bins

# %%
class EmptyBinException(Exception):
    """Custom exception for empty bins."""
    pass

def fit_xd_bin(subset, n_bins=100, n_trials=100):
    counts, bin_edges = np.histogram(subset['x_D'], bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    errors = np.sqrt(counts)

    # Normalize counts and errors
    total = np.sum(counts)
    counts_norm = counts / (bin_widths * total)  
    errors_norm = errors / (bin_widths * total)  

    # Mask: counts > 5 and density > 10⁻³
    print("Number of bins before mask :", len(bin_centers))
    mask = (counts > 5) & (counts_norm > 0.001) & (errors_norm < 0.5*counts_norm)
    bin_centers_masked = bin_centers[mask]
    print("Number of bins remaining after mask :", len(bin_centers_masked))
    if len(bin_centers_masked) == 0:
        raise EmptyBinException("No bins remaining after masking. Fit cannot proceed.")
    counts_norm_masked = counts_norm[mask]
    errors_norm_masked = errors_norm[mask]

    # --- Bounds (to ensure positive widths and normalizations) ---
    #   x0, xbarL, xbarR, sigmaL, gammaL, sigmaI, gammaI, sigmaR, gammaR, NL, NI, NR
    bounds = (
        [-20, -15, -3,  0.001, 0.01,  1e-8, 0.01,  0.01, 0.4,  0],
        [ 20, 0, 1,  13,  10,  6, 10, 6, 10, 100]
    )

    norm_array = bin_widths[mask] * total

    # Fit
    try:
        popt, fit_vals, chi_squared = random_start_random_walk_fit(
        D,
        bin_centers_masked,
        counts_norm_masked,
        errors_norm_masked,
        bounds,
        norm_array,
        noise_scale=0.2,  # Increase if still stuck
        verbose=False,
        n_trials=n_trials,  # Increase if still stuck
    )

    except RuntimeError as e:
        print("Fit failed:", e)
        popt = None

    # Compute chi-squared
    
    ndof = len(bin_centers_masked) - len(popt)  # number of degrees of freedom
    p_value = 1 - chi2.cdf(chi_squared, ndof)


    return popt, chi_squared, p_value, bin_centers_masked, counts_norm_masked, errors_norm_masked



# %%
# Fit and Plot each bin
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
fig.suptitle("Scaled Energy Deviation x_D in Each (E_bin, θ_bin)", fontsize=16)

for E_bin in range(n_E_bins):
    for theta_bin in range(n_theta_bins):
        print(f'Processning E_bin={E_bin}, theta_bin={theta_bin}')
        n_bins = 100
        subset = df[(df['E_bin'] == E_bin) & (df['theta_bin'] == theta_bin)]
        
        try:
            popt, chi_squared, p_value, bin_centers_masked, counts_norm_masked, errors_norm_masked = fit_xd_bin(subset, n_bins, n_trials = 30)
            print(f"Best popt : {popt}, len(opt) : {len(popt)}")
            np.save(resultPath + f"fit_params_LET_E{E_bin}_theta{theta_bin}.npy", popt)
        except EmptyBinException as e:
            print(e)
            popt = None
        # Save parameters


        if popt is not None:
            x0, xbarL, xbarR, sigmaL, gammaL, sigmaI, gammaI, sigmaR, gammaR, N = popt
            NL = N * R(xbarL, x0, sigmaI, gammaI) / R(xbarL, x0, sigmaL, gammaL)
            NR = N * R(xbarR, x0, sigmaI, gammaI) / R(xbarR, x0, sigmaR, gammaR)

            # Compute chi-squared
            ndof = len(bin_centers_masked) - len(popt)  # number of degrees of freedom

            chi_squared_str = f"$\chi^2$/ndof = {chi_squared:.2f}/{ndof} = {chi_squared/ndof:.2f}"
            print(f"P-value : ", p_value)
            if p_value < 1e-16:
                prob_str = "p-value < 1e-16"
            else:
                prob_str = f"p-value = {p_value:.2g}"

            ax = axes[E_bin, theta_bin]

            
            ax.errorbar(bin_centers_masked, counts_norm_masked, yerr=errors_norm_masked, fmt='o', color='royalblue', alpha=0.7, label="Data", markersize=3)
            ax.set_title(f"E_bin={E_bin}, θ_bin={theta_bin}", fontsize=9)
            
            x_fit = np.linspace(np.min(bin_centers_masked), np.max(bin_centers_masked), 1000)
            y_fit = D(x_fit, *popt)
            ax.plot(x_fit, y_fit, color='darkred', lw=2, label=f'Fit\n{chi_squared_str}\n{prob_str}')
            # Show components
            mask_L = x_fit - x0 < xbarL
            mask_I = (x_fit - x0 >= xbarL) & (x_fit - x0 <= xbarR)
            mask_R = x_fit - x0 > xbarR
            ax.plot(x_fit[mask_L], NL * R(x_fit[mask_L], x0, sigmaL, gammaL), color='green', label='Left')
            ax.plot(x_fit[mask_I], N * R(x_fit[mask_I], x0, sigmaI, gammaI), color='blue', label='Core')
            ax.plot(x_fit[mask_R], NR * R(x_fit[mask_R], x0, sigmaR, gammaR), color='orange', label='Right')


            ax.set_xlabel("$x_D$")
            ax.set_ylabel("$D(x_D)$")
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 2)
            ax.grid()
            ax.legend()
        else:
            ax = axes[E_bin, theta_bin]
            ax.set_visible(False)
            print(f"Fit failed for E_bin={E_bin}, theta_bin={theta_bin}")


plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(figPath + "xD_allBins_fits.png", dpi=400)
plt.show()

# %%
fig.savefig(figPath + "xd_HET_fit_all_bins.png", dpi=400)




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

if __name__ == "__main__":
    dirDir= "/home/users/l/lebrun/"
    dataPath = "/dpnc/beegfs/dampe/tryf/ForKimberly/"
    resultPath = dirDir + "results/"
    figPath = dirDir + "figures/"
    dataPath = dataPath +  "tt_filtered_cos_theta_bins/"

    try :
        os.mkdir(figPath+"Edist")
        os.mkdir(resultPath+"Edist")
    except FileExistsError:
        pass

    exponent = 0
    figPath = figPath+f"Edist/exp{exponent:.1f}/"
    resultPath = resultPath + f"Edist/exp{exponent:.1f}/"
    dataPath = dataPath + f"exp{exponent:.1f}/"

    files = browse_files(dataPath + "*.root")

    rdf = ROOT.RDataFrame("MC", files)

    df = rdf.AsNumpy(["tt_theta", "tt_ekin", "tt_bgoTotalEcorr_GeV", "weight"])
    df = pd.DataFrame(df)

    df = preprocess_df(df)

    # Compute energy relative difference
    df['Ediff'] = (df['E_reco'] - df['E_true'])/df['E_true']