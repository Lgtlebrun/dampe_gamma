import glob
import numpy as np
import os, sys
from matplotlib import pyplot as plt

from astropy.coordinates import SkyCoord
from astropy import units as u

PATH_GIEM_BINS = "/home/users/l/lebrun/results/all_bins_GIEM.npz"

BINS_NPZ = np.load(PATH_GIEM_BINS)
ENERGY_BINS_EDGES = BINS_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(ENERGY_BINS_EDGES[:-1] * ENERGY_BINS_EDGES[1:])  # GeV
def parse_par_file(filename):
    """
    Parse a .par file and extract RAJ and DECJ values.
    
    Parameters:
    filename (str): Path to the .par file
    
    Returns:
    tuple: (raj_value, decj_value) or (None, None) if not found
    """
    raj_value = None
    decj_value = None
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split line into components
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                parameter = parts[0].upper()
                
                # Extract RAJ value
                if parameter == 'RAJ':
                    # RAJ value is typically in format HH:MM:SS.ssss
                    raj_value = parts[1]
                
                # Extract DECJ value  
                elif parameter == 'DECJ':
                    # DECJ value is typically in format +/-DD:MM:SS.ssss
                    decj_value = parts[1]
    
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        return None, None
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None, None
    
    return raj_value, decj_value

def convert_ra_to_decimal(ra_str):
    """
    Convert RA string from HH:MM:SS.ssss format to decimal hours.
    
    Parameters:
    ra_str (str): RA in HH:MM:SS.ssss format
    
    Returns:
    float: RA in decimal hours
    """
    if ra_str is None:
        return None
    
    try:
        parts = ra_str.split(':')
        hours = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0
        seconds = float(parts[2]) if len(parts) > 2 else 0
        
        decimal_hours = hours + minutes/60.0 + seconds/3600.0
        return decimal_hours
    except (ValueError, IndexError):
        print(f"Error parsing RA: {ra_str}")
        return None

def convert_dec_to_decimal(dec_str):
    """
    Convert DEC string from +/-DD:MM:SS.ssss format to decimal degrees.
    
    Parameters:
    dec_str (str): DEC in +/-DD:MM:SS.ssss format
    
    Returns:
    float: DEC in decimal degrees
    """
    if dec_str is None:
        return None
    
    try:
        # Handle sign
        sign = 1
        if dec_str.startswith('+'):
            dec_str = dec_str[1:]
        elif dec_str.startswith('-'):
            sign = -1
            dec_str = dec_str[1:]
        
        parts = dec_str.split(':')
        degrees = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0
        seconds = float(parts[2]) if len(parts) > 2 else 0
        
        decimal_degrees = degrees + minutes/60.0 + seconds/3600.0
        return sign * decimal_degrees
    except (ValueError, IndexError):
        print(f"Error parsing DEC: {dec_str}")
        return None


def get_target_coords(par_dir):
    # Load target coordinates from .par
    filename = None
    try:
        for file in os.listdir(par_dir):
            if file.endswith(".par"):
                filename = os.path.join(par_dir, file)
                break
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No .par file found in {par_dir}. Target: {target}")

    if filename is None:
        raise FileNotFoundError(f"No .par file found for target {target}")

    raj_str, decj_str = parse_par_file(filename)
    if raj_str is None or decj_str is None:
        raise ValueError(f"Could not find RAJ and DECJ in {filename}")

    ra_target = convert_ra_to_decimal(raj_str)
    dec_target = convert_dec_to_decimal(decj_str)
    c = SkyCoord(ra=ra_target*u.hourangle, dec=dec_target*u.degree, frame='icrs').transform_to('galactic')
    target_lon = c.l.wrap_at(180 * u.deg).degree
    target_lat = c.b.degree

    return target_lon, target_lat

def convert_longitude(lon_edges):
    """Convert longitudes from [0, 2π] to [-π, π] for Mollweide."""
    return -np.where(lon_edges > np.pi, lon_edges - 2 * np.pi, lon_edges)


def plot_exposure_map(lon_centers, lat_centers, exposure_map, target_lon, target_lat, target_name="Target", figPath=None):
    """ Plot the cropped exposure map in rectangular projection.
    Args:
        lon_centers (np.ndarray): 1D array of galactic longitudes in radians.
        lat_centers (np.ndarray): 1D array of galactic latitudes in radians.
        exposure_map (np.ndarray): 2D array of exposure values.
        figPath (str, optional): Path to save the figure. If None, the figure is shown.
    """

    # Set global font size
    plt.rcParams.update({'font.size': 16})

    # Convert radians to degrees for plotting
    lon_deg = np.rad2deg(lon_centers)
    lat_deg = np.rad2deg(lat_centers)

    # 2D grid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg, indexing='ij')

    # Transpose for imshow-like orientation
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(lon_grid, lat_grid, exposure_map, cmap='viridis', shading='auto')
    ax.plot(target_lon, target_lat, 'rx', markersize=12, label=f'{target_name}')
    ax.xaxis.set_inverted(True)  # Invert x-axis for galactic longitude
    ax.legend(loc='upper right', fontsize=16)

    ax.set_xlabel('Galactic Longitude [deg]', fontsize=18)
    ax.set_ylabel('Galactic Latitude [deg]', fontsize=18)
    ax.set_title(f'Exposure Map of {target_name} ROI at {energy:.3f} GeV', fontsize=20)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Exposure [cm² s]', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_aspect('auto')
    ax.grid(True)

    if figPath:
        plt.savefig(figPath)
        plt.close(fig)
    else:
        plt.show()

def sum_exposure_maps(result_dir, output_path="summed_exposure_map.npz"):
    """
    Sum all exposure maps found in result_dir and save the total map.
    """
    print("Starting to sum exposure maps...")
    summed_map = None
    map_count = 0

    # Scan result_dir for exposure map files (adjust pattern as needed)
    exposure_files = sorted(glob.glob(os.path.join(result_dir, "*.npz")))

    if not exposure_files:
        print("No exposure maps found to sum.")
        return
    print(f"Found {len(exposure_files)} exposure map files in {result_dir}")
    print(f"DEBUG : exposure_files[0] = {exposure_files[0]}")

    with np.load(exposure_files[0]) as ds:
        lon_edges = ds["lon_edges"]
        lat_edges = ds["lat_edges"]
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    for file in exposure_files:
        data = np.load(file)
        exposure_map = data["exposure_map"]  # adjust key if needed

        if summed_map is None:
            summed_map = np.zeros_like(exposure_map)
        
        summed_map += exposure_map
        map_count += 1

    # Save the total summed map
    np.savez_compressed(output_path, exposure_map=summed_map, lon_edges=lon_edges, lat_edges=lat_edges)

    print(f"Summed {map_count} exposure maps.")
    print(f"Total summed map saved to {output_path}")

    return summed_map, lon_centers, lat_centers


if __name__ == "__main__":

    energy_bin = int(sys.argv[1])
    target = sys.argv[2]

    energy = ENERGY_BINS_CENTERS[energy_bin] 

    base_dir = "/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2"
    result_dir = f"/dpnc/beegfs/dampe/tryf/ForKimberly/exposureMaps_targets/{target}/"
    fig_dir = "/home/users/l/lebrun/figures"
    par_dir = f"/home/users/l/lebrun/data/{target}"

    daily_result_dir = f"{result_dir}/{energy:.3f}GeV/daily/"
    output_path = f"{result_dir}/{energy:.3f}GeV/summed_exposure_map.npz"
    expoMap, lon_centers, lat_centers = sum_exposure_maps(daily_result_dir, output_path)

    print(f"DEBUG : lon_centers, lat_centers = {lon_centers}, {lat_centers}")

    # lon_centers = convert_longitude(lon_centers)

    print(f"DEBUG : lon_centers after conversion = {lon_centers}, lat_centers = {lat_centers}")

    target_lon, target_lat = get_target_coords(par_dir)
    # target_lon = convert_longitude(target_lon)

    plot_exposure_map(lon_centers, lat_centers, expoMap, target_lon, target_lat, target_name=target,
                      figPath=f"{fig_dir}/{target}/exposureMaps/{energy:.3f}GeV/summed_exposure_map.png")
    print(f"Exposure map for {target} at {energy:.3f} GeV saved to {fig_dir}/{target}/exposureMaps/{energy:.3f}GeV/summed_exposure_map.png")