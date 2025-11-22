import os
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 300

# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
PATH_BINS = "/home/users/l/lebrun/results/effArea_bins_info_HET.npz"
# Load the effective area table
EFF_AREA_NPZ = np.load(PATH_EFFAREA)

# Get bins edges and centers
ENERGY_BINS_EDGES = EFF_AREA_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(ENERGY_BINS_EDGES[:-1] * ENERGY_BINS_EDGES[1:])  # GeV
COSTHETA_BINS_EDGES = EFF_AREA_NPZ['theta_bins_edge']
PHI_BINS_EDGES = EFF_AREA_NPZ['phi_bins_edge']

def get_exposure_files(data_dir):
    """Get list of all .npz exposure map files."""
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    print(f"Found {len(files)} files.")
    return sorted(files)

def get_exposure_sum(files):
    """Sum exposure maps one by one to reduce memory usage."""
    summed_map = None

    for i, file in enumerate(files):
        try:
            with np.load(file) as ds:
                exposure_map = ds["exposure_map"]

                if summed_map is None:
                    summed_map = np.zeros_like(exposure_map)

                summed_map += exposure_map

            if i % 50 == 0:  # Optional: progress every 50 files
                print(f"Summed {i+1}/{len(files)} maps")

        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    return summed_map


def plot_exposure_map(lon_centers, lat_centers, exposure_map, energy, fig_path=None, title="Exposure Map"):
    """Plot the exposure map in Mollweide projection."""
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')

    # Prepare for Mollweide projection
    lon_plot = lon_grid.T
    lat_plot = lat_grid.T
    exposure_plot = exposure_map.T

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    mesh = ax.pcolormesh(lon_plot, lat_plot, exposure_plot, cmap='viridis', shading='auto')

    fig.colorbar(mesh, ax=ax, label='Exposure [cm² sr s]')
    ax.set_title(f"{title} ({energy:.3f} GeV)")

    tick_labels = np.array([150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150])
    ax.set_xticklabels(tick_labels)
    ax.xaxis.set_tick_params(labelcolor="white")
    ax.set_xlabel('Galactic Longitude')
    ax.set_ylabel('Galactic Latitude')
    ax.grid(True)

    if fig_path:
        fig.savefig(fig_path)
        print(f"Saved plot to {fig_path}")
    else:
        plt.show()
    
    return fig

def convert_longitude(lon_edges):
    """Convert longitudes from [0, 2π] to [-π, π] for Mollweide."""
    return -np.where(lon_edges > np.pi, lon_edges - 2 * np.pi, lon_edges)

def save_exposure_map(result_path, lon_edges, lat_edges, exposure_map):
    """Save exposure map to disk."""
    np.savez(result_path, lon_edges=lon_edges, lat_edges=lat_edges, exposure_map=exposure_map)
    print(f"Saved exposure map to {result_path}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_total_exposure_map.py Energy[GeV]")
        sys.exit(1)

    energy_bin = int(sys.argv[1])
    energy = ENERGY_BINS_CENTERS[energy_bin]  # Convert energy bin to GeV

    dirDir = "/home/users/l/lebrun/"
    data_dir = f"/dpnc/beegfs/dampe/tryf/ForKimberly/exposureMaps_GIE/{energy:.3f}GeV/"
    data_dir_daily = data_dir + "daily/"
    fig_dir = os.path.join(dirDir, "figures", "exposureMaps_GIE")
    result_dir = os.path.join(dirDir, "results", "exposureMaps")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    fig_path = os.path.join(fig_dir, f"{energy:.3f}GeV_exposure_sum.png")
    result_path = os.path.join(result_dir, f"{energy:.3f}GeV_exposure_sum.npz")
    result_path_beegfs = os.path.join(data_dir, f"{energy:.3f}GeV_exposure_sum.npz")

    print(f"Loading exposure maps from {data_dir}")
    files = get_exposure_files(data_dir_daily)

    with np.load(files[0]) as ds:
        lon_edges = ds["lon_edges"]
        lat_edges = ds["lat_edges"]
        
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_edges_conv = convert_longitude(lon_edges)
    lon_centers_conv = convert_longitude(lon_centers)

    exposure_sum = get_exposure_sum(files)
    plot_exposure_map(lon_centers_conv, lat_centers, exposure_sum, energy, fig_path=fig_path)
    save_exposure_map(result_path, lon_edges_conv, lat_edges, exposure_sum)
    save_exposure_map(result_path_beegfs, lon_edges_conv, lat_edges, exposure_sum)

    print("Exposure map summation complete.")


