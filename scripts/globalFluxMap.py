import numpy as np
import pandas as pd
import ROOT
import glob
import sys, os
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from astropy_healpix import pixel_resolution_to_nside


# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
PATH_BINS = "/home/users/l/lebrun/results/effArea_bins_info_HET.npz"
# Load the effective area table
EFF_AREA_NPZ = np.load(PATH_EFFAREA)
# Get bins edges and centers
ENERGY_BINS_EDGES = EFF_AREA_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(ENERGY_BINS_EDGES[:-1] * ENERGY_BINS_EDGES[1:])  # GeV

PSF_path = "/home/users/l/lebrun/results/PSF/exp0.0/script/resolution_per_bin_w.csv"
PSF_infos_path = "/home/users/l/lebrun/results/PSF/exp0.0/script/infos_bins.npz"
df_PSF = pd.read_csv(PSF_path)
print(df_PSF.head())

PSF_infos = np.load(PSF_infos_path)
print([key for key in PSF_infos.keys()])


def find_PSF_Ebin(energy, infos) :
    E_edges = np.asarray(infos["E_bins_edge"])
    # searchsorted gives the position where energy would be inserted
    idx = np.searchsorted(E_edges, energy) - 1

    # Out-of-range protection
    if idx < 0 or idx >= len(E_edges) - 1:
        return -1  # or raise ValueError("Energy outside bins")

    return idx



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


def plot_hpx_map(hpx_map, 
                 title="", 
                 label="", 
                 output_name="", 
                 logNorm=False, 
                 figPath=None):

    # ---- Determine min/max safely ----
    positive = hpx_map[hpx_map > 0]

    if positive.size == 0:
        vmin, vmax = 1, 10  # fallback for empty map
    else:
        vmin = float(positive.min())
        vmax = float(positive.max())

        # avoid degenerate case
        if vmax <= vmin:
            vmax = vmin * 10

    # ---- Plot ----
    plt.figure(figsize=(10, 5))

    if logNorm:
        hp.mollview(
            hpx_map,
            title=title,
            unit=label,
            norm='log',        # <-- IMPORTANT
            min=vmin,
            max=vmax,
            cmap="viridis",
            cbar=True,
            notext=False,
            badcolor="black"
        )
    else:
        hp.mollview(
            hpx_map,
            title=title,
            unit=label,
            min=vmin,
            max=vmax,
            cmap="viridis",
            cbar=True,
            notext=False,
            badcolor="black"
        )
    # Add custom axes-like styling
    style_healpy_mollview(fontsize=9)

    # ---- Save ----
    if figPath:
        figName = figPath + output_name
        plt.savefig(figName, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {figName}")

    return plt.gcf()

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

def plot_counts_map_hpx(hpx_counts_map, energy=None, figPath=None):
    """
    Replacement for your plot_counts_map but using HEALPix.
    """

    energy_center = ENERGY_BINS_CENTERS[energy] if energy is not None else None
    energy_low = ENERGY_BINS_EDGES[energy] if energy is not None else None
    energy_high = ENERGY_BINS_EDGES[energy + 1] if energy is not None else None

    energy_low_str = f"{energy_low:.2f}"
    energy_high_str = f"{energy_high:.2f}"

    title = "γ-rays Counts, 9 years"
    label = "Counts"
    output_name = "CountMap"

    if energy:
        title += f" | Energy : [{energy_low_str} - {energy_high_str}] GeV"
        #title += f" | Energy : 1 GeV - 10 TeV"        
        output_name += f"-{energy}"

    output_name += ".png"

    return plot_hpx_map(
        hpx_counts_map,
        title=title,
        label=label,
        output_name=output_name,
        logNorm=True,
        figPath=figPath
    )

def plot_flux_map(lon_centers, lat_centers, flux_map, energy=None, figPath=None):
    title = "γ-rays Flux, 9 years"
    label = "Counts/Exposure [cm⁻2 s-1]"
    output_name = "FluxMap"
    if energy:
        title += f" | Energy : {energy} GeV"
        output_name += f"-{energy}"
    output_name += ".png"
    return plot_map(lon_centers, lat_centers, flux_map, title, label, output_name, True, figPath)

def style_healpy_mollview(fontsize=10):
    """
    Reproduce axis-like ticks and labels on healpy mollview plots.
    """
    # Draw graticule lines every 30°
    hp.graticule(dpar=30, dmer=30, alpha=0.5, color="white")

    # Add manual labels at selected longitudes
    for lon_deg in np.arange(-150, 151, 30):
        hp.projtext(
            lon_deg, 
            0, 
            f"{lon_deg}°",
            lonlat=True,
            color="white",
            fontsize=fontsize,
            ha="center",
            va="center"
        )

    # Add manual labels at selected latitudes
    for lat_deg in np.arange(-75, 76, 15):
        hp.projtext(
            0,
            lat_deg,
            f"{lat_deg}°",
            lonlat=True,
            color="white",
            fontsize=fontsize,
            ha="center",
            va="center"
        )

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

    PSF_Ebin = find_PSF_Ebin(energy, PSF_infos)
    print(f"Energy : {energy} GeV, PSF ang resolution bin : {PSF_Ebin}")
    ang_resolution = df_PSF.angular_resolution[df_PSF.E_bin == 0].max()
    print(f"Angular resolution from PSF : {ang_resolution}")
    NSIDE = pixel_resolution_to_nside(ang_resolution * u.deg)
    NPIX = hp.nside2npix(NSIDE)
    print(
        "Approximate resolution at NSIDE {} is {:.2} deg".format(
            NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
        )
    )   

    figPath += energy_exposure_map_str + "GeV/"
    try : 
        if not os.path.exists(figPath):
            os.makedirs(figPath)
    except FileExistsError as e:
        pass

    df = load_and_process_df(dataFile)

    # Build HEALPix count map
    NSIDE = 64
    NPIX = hp.nside2npix(NSIDE)

    lon = df['lon'].values
    lat = df['lat'].values
    theta = np.pi/2 - lat
    phi = lon

    ipix = hp.ang2pix(NSIDE, theta, phi)

    count_map = np.zeros(NPIX, dtype=float)
    np.add.at(count_map, ipix, 1)

    # Plot with HEALPix
    figName = figPath + "CountMapHealpix"
    fig = plot_counts_map_hpx(count_map, energy=energy_bin, figPath=figName)