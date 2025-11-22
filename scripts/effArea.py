
from multiprocessing import process
import ROOT, glob, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

cos_theta_min = 0.5
cos_theta_max = 1
phi_min = 0
phi_max = 360
n_theta_bins = 10
n_phi_bins = 360

Emin = 1
Emax_HET = 1e4
n_E_bins_HET = 40 # 40

E_bins_edge_HET = np.logspace(np.log10(Emin), np.log10(Emax_HET), n_E_bins_HET + 1)
E_bins_center_HET = np.sqrt(E_bins_edge_HET[:-1] * E_bins_edge_HET[1:])

def calculate_bin_edges(centers):
    """Calculate bin edges from bin centers assuming log spacing"""
    edges = []
    
    # First edge: extrapolate backwards
    if len(centers) > 1:
        ratio_0_1 = centers[1] / centers[0]
        first_edge = centers[0] / np.sqrt(ratio_0_1)
    else:
        first_edge = centers[0] / 2  # fallback
    edges.append(first_edge)
    
    # Middle edges: geometric mean of adjacent centers
    for i in range(len(centers) - 1):
        edge = np.sqrt(centers[i] * centers[i + 1])
        edges.append(edge)
    
    # Last edge: extrapolate forwards
    if len(centers) > 1:
        ratio_last = centers[-1] / centers[-2]
        last_edge = centers[-1] * np.sqrt(ratio_last)
    else:
        last_edge = centers[-1] * 2  # fallback
    edges.append(last_edge)
    
    return np.array(edges)

theta_bins_edge = np.linspace(cos_theta_min, cos_theta_max, n_theta_bins + 1)
theta_bins_center = (theta_bins_edge[:-1] + theta_bins_edge[1:])/2

phi_bins_edge = np.linspace(phi_min, phi_max, n_phi_bins + 1)
phi_bins_center = (phi_bins_edge[:-1] + phi_bins_edge[1:])/2

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

def process_df(df):
    
    df['E_true'] = df['tt_ekin'] * 1e-3 # Convert tt_ekin from MeV to GeV
    # Safeguard : clip where theta in [-pi, pi]
    df = df.loc[(df['tt_theta'] < np.pi) & (df['tt_theta'] > -np.pi)]
    df = df.rename(columns={'tt_theta' : 'theta_True', 'tt_phi' : 'phi_True'})

    df['theta_True_deg']= np.rad2deg(df['theta_True'])
    df['phi_True_deg'] = np.rad2deg(np.mod(df['phi_True'], 2 * np.pi))

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
    
    if "phi_True_deg" in df.columns:
        print(f"Minimum phi_True_deg: {df['phi_True_deg'].min():.2f} degrees")
        print(f"Maximum phi_True_deg: {df['phi_True_deg'].max():.2f} degrees")
        print(f"Mean phi_True_deg: {df['phi_True_deg'].mean():.2f} degrees")

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

def compute_efficiency(df, dfHET):
    # Create 2D histograms using numpy
    # Total events (unweighted)
    total_counts, edges = np.histogramdd(
        np.column_stack([df['E_true'], df['cosTheta'], df['phi_True_deg']]),
        bins=[E_bins_edge_HET, theta_bins_edge, phi_bins_edge]
    )

    # Accepted events (unweighted)
    accepted_counts, _= np.histogramdd(
        np.column_stack([dfHET['E_true'], dfHET['cosTheta'], dfHET['phi_True_deg']]), 
        bins=[E_bins_edge_HET, theta_bins_edge, phi_bins_edge]
    )

    # Total events (weighted)
    total_weights, _ = np.histogramdd(
        np.column_stack([df['E_true'], df['cosTheta'], df['phi_True_deg']]),
        bins=[E_bins_edge_HET, theta_bins_edge, phi_bins_edge],
        weights=df['weight']
    )

    # Accepted events (weighted)
    accepted_weights, _ = np.histogramdd(
        np.column_stack([dfHET['E_true'], dfHET['cosTheta'], dfHET['phi_True_deg']]), 
        bins=[E_bins_edge_HET, theta_bins_edge, phi_bins_edge],
        weights=dfHET['weight']
    )


    # Total events (unweighted)
    total_2D, _, _ = np.histogram2d(
        df['E_true'], df['cosTheta'],
        bins=[E_bins_edge_HET, theta_bins_edge]
    )

    # Accepted events (unweighted)
    accepted_2D, _, _ = np.histogram2d(
        dfHET['E_true'], dfHET['cosTheta'],
        bins=[E_bins_edge_HET, theta_bins_edge]
    )

    # Total events (weighted)
    total_2D_w, _, _ = np.histogram2d(
        df['E_true'], df['cosTheta'],
        bins=[E_bins_edge_HET, theta_bins_edge],
        weights=df['weight']
    )

    # Accepted events (weighted)
    accepted_2D_w, _, _ = np.histogram2d(
        dfHET['E_true'], dfHET['cosTheta'],
        bins=[E_bins_edge_HET, theta_bins_edge],
        weights=dfHET['weight']
    )

    # Total events (unweighted)
    total_1D, _ = np.histogram(
        df['E_true'], bins=E_bins_edge_HET
    )

    # Accepted events (unweighted)
    accepted_1D, _ = np.histogram(
        dfHET['E_true'], bins=E_bins_edge_HET
    )

    # Total events (weighted)
    total_1D_w, _ = np.histogram(
        df['E_true'], bins=E_bins_edge_HET, weights=df['weight']
    )

    # Accepted events (weighted)
    accepted_1D_w, _ = np.histogram(
        dfHET['E_true'], bins=E_bins_edge_HET, weights=dfHET['weight']
    )

    # Calculate ratios (with protection against division by zero)
    unweighted_ratio = np.divide(accepted_counts, total_counts, 
                            out=np.zeros_like(accepted_counts), 
                            where=total_counts!=0)

    weighted_ratio = np.divide(accepted_weights, total_weights, 
                            out=np.zeros_like(accepted_weights), 
                            where=total_weights!=0)
    

    unweighted_ratio_2D = np.divide(accepted_2D, total_2D, 
                            out=np.zeros_like(accepted_2D, dtype=float), 
                            where=total_2D!=0)

    weighted_ratio_2D = np.divide(accepted_2D_w, total_2D_w, 
                            out=np.zeros_like(accepted_2D_w, dtype=float), 
                            where=total_2D_w!=0)
    
    unweighted_ratio_1D = np.divide(accepted_1D, total_1D,
                            out=np.zeros_like(accepted_1D, dtype=float), 
                            where=total_1D!=0)
    weighted_ratio_1D = np.divide(accepted_1D_w, total_1D_w,
                            out=np.zeros_like(accepted_1D_w, dtype=float), 
                            where=total_1D_w!=0)
                        
    # Calculate binomial errors for efficiency (use unweighted counts for errors)
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.sqrt(unweighted_ratio_1D * (1 - unweighted_ratio_1D) / total_1D)
        err = np.nan_to_num(err)  # replace nan with 0 (if total_counts=0)
    

    return unweighted_ratio, weighted_ratio, unweighted_ratio_2D, weighted_ratio_2D, unweighted_ratio_1D, weighted_ratio_1D, err

def plot_effArea(effArea, figName, weighted = False) :

    title = "Effective Area"
    if weighted :
        title = "Weighted " + title

    fig, ax = plt.subplots()

    effArea = effArea.T # Transpose to match the shape of E_bins_edge_HET and theta_bins_edge

    c1 = ax.pcolormesh(
        E_bins_edge_HET, theta_bins_edge, effArea,
        shading='auto', cmap='viridis'
    )
    ax.set_xscale('log')
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("cos(θ)")
    ax.set_title(title)
    fig.colorbar(c1, ax=ax, label='Effective Area [cm²]')

    plt.tight_layout()
    fig.savefig(figName)

def plot_efficiency(efficiency, err=None, figName=None):
    
    plt.figure(figsize=(8,5))
    if err is not None :
        plt.errorbar(E_bins_center_HET, efficiency*100,yerr=err*100, fmt='o-', color='green',
                    ecolor='green', capsize=3, label='Binomial errors')
    else:
        plt.plot(E_bins_center_HET, efficiency*100, 'o-', color='green', label='Efficiency')
    plt.xscale('log')
    plt.xlabel("Energy [GeV]") 
    plt.ylabel("Efficiency (accepted / total) [%]")
    plt.title("Detection Efficiency vs Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if figName is not None:
        plt.savefig(figName)

    

def save_effArea(data, fileName) :
    np.savez(fileName, E_bins_edge = E_bins_edge_HET, theta_bins_edge = theta_bins_edge, phi_bins_edge = phi_bins_edge , effArea = data)


if __name__ == "__main__":

    dirDir= "/home/users/l/lebrun/"
    dataDir = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma/"
    figPath_ = dirDir + "figures/effArea/script/"
    resultPath = dirDir + "results/effArea/"
    dataPath1 = dirDir + "data/"
    dataPath2 = "/dpnc/beegfs/dampe/tryf/ForKimberly/"

    exponent = 0
    treename = "MC"

    figPath = figPath_+f"exp{exponent:.1f}/"

    dataPath_scaled = dataPath2 + "tt_scaled" + "_cos_theta_bins/"
    dataPath_scaled += f"exp{exponent:.1f}/"

    print(f"Looking for total files in {dataPath_scaled}...")
    files_scaled = browse_files(dataPath_scaled + "*.root")
    print(f"Files found : {files_scaled}")

    dataPath_filtered = dataPath2 + "tt_filtered" + "_cos_theta_bins/"
    dataPath_filtered += f"exp{exponent:.1f}/"

    print(f"Looking for total files in {dataPath_filtered}...")
    files_filtered = browse_files(dataPath_filtered + "*.root")
    print(f"Files found : {files_filtered}")

    # Load and process
    print("Loading Dataframes...")

    rdf = ROOT.RDataFrame(treename, files_scaled)
    print("DEBUG : loaded rdf")
    rdf_filtered = ROOT.RDataFrame(treename, files_filtered)
    print("DEBUG : loaded rdf_filtered")



    print("Converting rdf to pandas...")
    df = rdf.AsNumpy(["tt_ekin", "weight", "tt_theta", "tt_phi", "theta_BestSTK", 'tt_bgoTotalEcorr_GeV', "tt_isHET"])
    df=pd.DataFrame(df)
    print("DEBUG : converted rdf to df")

    print("Converting rdf_filtered to pandas...")
    df_filtered = rdf_filtered.AsNumpy(["tt_ekin", "weight", "tt_theta", "tt_phi", "theta_BestSTK", 'tt_bgoTotalEcorr_GeV', "tt_isHET"])
    df_filtered = pd.DataFrame(df_filtered)
    print("DEBUG : converted rdf_filtered to df_filtered")
    print(f"DEBUG : number of entries in df = {len(df)}")
    print(f"DEBUG : number of entries in df_filtered = {len(df_filtered)}")

    df = process_df(df)
    df_filtered = process_df(df_filtered)

    # Print infos

    infos_df(df, "Total Events")
    infos_df(df_filtered, "Filtered events")

    print("Computing efficiencies...")
    unweighted_ratio, weighted_ratio, unweighted_ratio_int_phi, weighted_ratio_int_phi, unweighted_ratio_1D, weighted_ratio_1D, err  = compute_efficiency(df, df_filtered)

    r = 1.38*1e2 # cm
    phi_norm = 2*np.pi # used to average over azimuthal angle

    df_results_geom = 37.59 * 1e4 # cm^2 sr
    cross_section = 6 # m^2
    eff_area_geom = 6*1e4 # cm^2

    effArea = unweighted_ratio * eff_area_geom
    effArea_w = weighted_ratio * eff_area_geom

    effArea_averaged_w = np.mean(effArea_w, axis=2)

    effArea_2D = unweighted_ratio_int_phi * eff_area_geom / phi_norm
    effArea_w_2D = weighted_ratio_int_phi * eff_area_geom / phi_norm


    # TODO : assess and remove/save
    effArea_int_phi = effArea_2D * phi_norm


    fileName = "effectiveArea_unweighted"
    fileName_w = "effectiveArea_weighted"
    fileName_av_phi = "effectiveArea_weighted_averaged_phi"

    try : 
        if not os.path.exists(figPath):
            os.makedirs(figPath)
    except FileExistsError as e:
        pass

    try : 
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
    except FileExistsError as e:
        pass

    figName = figPath + f"{fileName}.png"
    figName_w = figPath + f"{fileName_w}.png"

    plot_effArea(effArea_2D, figName)
    plot_effArea(effArea_w_2D, figName_w, True)
    plot_effArea(effArea_averaged_w, figPath + f"{fileName_av_phi}.png", True)
    # TODO assess and remove
    plot_effArea(effArea_int_phi, figPath + f"phi_int_{fileName}.png")

    plot_efficiency(unweighted_ratio_1D, err, figPath + f"efficiency.png")
    plot_efficiency(weighted_ratio_1D, figName = figPath + f"efficiency_weighted.png")


    print(f"EffAreas plotted to {figName} and {figName_w}")

    save_effArea(effArea, resultPath + fileName + ".npz")
    save_effArea(effArea_w, resultPath + fileName_w + ".npz")

    print(f"EffAreas saved to {fileName} and {fileName_w}")
