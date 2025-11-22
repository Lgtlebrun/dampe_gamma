import numpy as np
import pandas as pd
import ROOT
import glob
import sys, os
import matplotlib.pyplot as plt

# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
PATH_BINS = "/home/users/l/lebrun/results/effArea_bins_info_HET.npz"
# Load the effective area table
EFF_AREA_NPZ = np.load(PATH_EFFAREA)
# Get bins edges and centers
ENERGY_BINS_EDGES = EFF_AREA_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(ENERGY_BINS_EDGES[:-1] * ENERGY_BINS_EDGES[1:])  # GeV

# Grid definition
lon_edges = np.linspace(-np.pi, np.pi, 720 + 1)
lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
dlon = lon_edges[1] - lon_edges[0]

lat_centers = np.linspace(-np.pi/2, np.pi/2, 361)
dlat = lat_centers[1] - lat_centers[0]
lat_edges = np.zeros(len(lat_centers) + 1)
lat_edges[0] = lat_centers[0] - dlat
lat_edges[1:] = lat_centers


def load_and_process_df(dataFile):
    tree = None
    f = ROOT.TFile(dataFile, "READ") 

    tree = f.Get("DmlNtup")
    print(tree)
    print(type(tree))
    rdf = ROOT.RDataFrame(tree)
    df = rdf.AsNumpy(["celestialcoordBest_2", "celestialcoordBest_3", "tt_bgoTotalEcorr_GeV"])
    df = pd.DataFrame(df)

    df = df.rename(columns={"celestialcoordBest_2" : "lon", "celestialcoordBest_3" : "lat", 
                            "tt_bgoTotalEcorr_GeV" : "energy"})

    df['lon_deg'] = df['lon']*180/np.pi
    df['lat_deg'] = df['lat']*180/np.pi
    df['lon'] = np.where(df['lon'] > np.pi, df['lon'] - 2*np.pi, df['lon'])

    return df

import matplotlib.colors as mcolors
def plot_map(lon_centers, lat_centers, map, title="", label="", output_name="", logNorm = False, figPath = None):
    """ Plot the exposure map in Mollweide projection.  
    Args:
        lon_centers (np.ndarray): 1D array of galactic longitudes in radians.
        lat_centers (np.ndarray): 1D array of galactic latitudes in radians.
        map (np.ndarray): 2D array of exposure values.
        figPath (str, optional): Path to save the figure. If None, the figure is shown.
    """

    # Convert your 1D radian lon/lat centers into 2D grid
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')

    # Convert to Mollweide's expected range: [-π, π] in lon, [-π/2, π/2] in lat
    lon_plot = lon_grid.T  # Transpose to match shape
    lat_plot = lat_grid.T
    map_plot = map.T  # Transpose as well to match pcolormesh layout

    # Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    # Plot with pcolormesh (lon, lat must be in radians)
    if logNorm :
        norm = mcolors.LogNorm()  # Use logarithmic normalization for color scale
    else : 
        norm = None
    mesh = ax.pcolormesh(lon_plot, lat_plot, map_plot, cmap='viridis', shading='auto', norm=norm)

    # Final touches
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_title(title)
    tick_labels = np.array([150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150])
    ax.set_xticklabels(tick_labels)
    ax.xaxis.set_tick_params(labelcolor="white")
    ax.set_xlabel('Galactic Longitude')
    ax.set_ylabel('Galactic Latitude')
    ax.set_facecolor('black')
    ax.grid(True)

    if figPath :
        figName = figPath + output_name
        fig.savefig(figName)

    return fig

def plot_exposure_map(lon_centers, lat_centers, exposure_map, energy, figPath = None):
    """ Plot the exposure map in Mollweide projection.  
    Args:
        lon_centers (np.ndarray): 1D array of galactic longitudes in radians.
        lat_centers (np.ndarray): 1D array of galactic latitudes in radians.
        exposure_map (np.ndarray): 2D array of exposure values.
        figPath (str, optional): Path to save the figure. If None, the figure is shown.
    """

    title = f'Exposure Map ({energy} GeV)'
    label = 'Exposure [cm² sr s]'
    output_name = f"{energy}GeV_exposure_sum.png"

    return plot_map(lon_centers, lat_centers, exposure_map, title, label, output_name, figPath)

def plot_counts_map(lon_centers, lat_centers, counts_map, energy = None, figPath = None):

    title = "γ-rays Counts, 9 years"
    label = "Counts"
    output_name = "CountMap"
    if energy:
        title += f" | Energy : {energy} GeV"
        output_name += f"-{energy}"
    output_name += ".png"

    return plot_map(lon_centers, lat_centers, counts_map, title, label, output_name, True, figPath)

def plot_flux_map(lon_centers, lat_centers, flux_map, energy=None, figPath=None):
    title = "γ-rays Flux, 9 years"
    label = "Counts/Exposure [cm⁻2 s-1]"
    output_name = "FluxMap"
    if energy:
        title += f" | Energy : {energy} GeV"
        output_name += f"-{energy}"
    output_name += ".png"
    return plot_map(lon_centers, lat_centers, flux_map, title, label, output_name, True, figPath)


if __name__ == "__main__":
    dirDir= "/home/users/l/lebrun/"
    resultPath = dirDir + "results/"
    figPath = dirDir + "figures/fluxMaps/"
    dataPath = dirDir + "data/"
    dataFile = dataPath + "DmlNtup_Data_9years_FullPhotonData_AllSky.root"
    exposurePath = "/dpnc/beegfs/dampe/tryf/ForKimberly/filtered_cos_theta_bins/exposureMaps/"

    energy_bin = 6
    energy = ENERGY_BINS_CENTERS[energy_bin]
    energy_exposure_map_str = f"{energy:.1f}"

    figPath += energy_exposure_map_str + "GeV/"
    try : 
        if not os.path.exists(figPath):
            os.makedirs(figPath)
    except FileExistsError as e:
        pass

    df = load_and_process_df(dataFile)

    hist, xedges, yedges = np.histogram2d(df['lon'], df['lat'], bins=[lon_edges, lat_edges])

    figName = figPath + f"CountMap"
    fig = plot_counts_map(np.sort(lon_centers), np.sort(lat_centers), hist, energy=energy_exposure_map_str, figPath=figName)
    print(f"Saved counts map to {figName}")