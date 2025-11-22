import numpy as np
import pandas as pd
import ROOT
import glob
import sys, os, re
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from astropy.coordinates import CartesianRepresentation, SkyCoord, angular_separation, EarthLocation, GCRS, ITRS
from astropy import units as u
from shapely.geometry import Point, Polygon
from shapely import vectorized
import psutil
import warnings
warnings.filterwarnings("ignore", message=".*np.bool.*", category=DeprecationWarning)

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

# SAA Region

saa_lat_2016 = [
    -8.136145, -5.522644, -2.909143, -0.295642, 2.317859, 4.931360,
    -1.170294, -5.924736, -10.080192, -13.636662, -16.594145, -18.952644,
    -32.560543, -34.555897, -36.591114, -38.661793, -40.743484, -42.791676,
    -44.741795, -46.509228, -47.989288, -49.057251, -49.568325, -49.357666,
    -48.240376, -46.011497, -42.446022, -37.298885, -30.304964, -21.179083
]

saa_lon_2016 = [
    -95.0, -84.0, -73.0, -62.0, -51.0, -40.0,
    -35.0, -29.0, -23.0, -17.0, -11.0, -5.0,
    -5.0, -10.0, -15.0, -20.0, -25.0, -30.0,
    -35.0, -40.0, -45.0, -50.0, -55.0, -60.0,
    -65.0, -70.0, -75.0, -80.0, -85.0, -90.0
]

saa_polygon_2016 = list(zip(saa_lon_2016, saa_lat_2016))
saa_polygon_shape = Polygon(saa_polygon_2016)  # Define SAA region

def met_to_utc(seconds, microseconds):
    seconds = np.asarray(seconds)
    microseconds = np.asarray(microseconds)
    delta_sec = TimeDelta(seconds, format='sec')
    delta_us = TimeDelta(microseconds * 1e-6, format='sec')
    epoch = Time("2013-01-01T00:00:00", scale="utc")
    return epoch + delta_sec + delta_us

def mask_saa(df):
    """
    Mask out timestamps where the satellite is within the SAA region.
    
    Parameters:
    - df: pandas.DataFrame with columns ['second', 'microsecond', 'x', 'y', 'z', 'utc']
    
    Returns:
    - df_masked: DataFrame excluding SAA timestamps
    - mask: boolean mask array (True = keep, False = SAA)
    """
    # Convert MET to UTC Time objects
    obstime = df['utc']

    # Cartesian coordinates in meters
    positions = CartesianRepresentation(
        x=df['x'].to_numpy() * u.m,
        y=df['y'].to_numpy() * u.m,
        z=df['z'].to_numpy() * u.m
    )

    # Build GCRS coordinate frame at each time
    gcrs_coords = GCRS(positions, obstime=obstime)

    # Convert to Earth-fixed frame (ITRS), then to geographic coordinates
    itrs_coords = gcrs_coords.transform_to(ITRS(obstime=obstime))
    geodetic = itrs_coords.earth_location.to_geodetic()

    # Extract lat/lon in degrees
    lat = geodetic.lat.to(u.deg).value
    lon = geodetic.lon.to(u.deg).value

    # Define SAA region
    mask_inside_saa = vectorized.contains(saa_polygon_shape, lon, lat)

    # Invert to create a mask of valid points
    valid_mask = ~mask_inside_saa
    df_masked = df[valid_mask].reset_index(drop=True)
    return df_masked, valid_mask

def load_filtered_day_dataframe(day_path, next_day_path, day_utc_start, day_utc_end):
    files = sorted(glob.glob(day_path) + glob.glob(next_day_path))
    rdf = ROOT.RDataFrame("HousekeepingData/SatStatus", files)
    data = rdf.AsNumpy(['DmpHKDSatStatus.fSecond', 'DmpHKDSatStatus.fMicrosecond',
                        'DmpHKDSatStatus.fPosition.fX', 'DmpHKDSatStatus.fPosition.fY', 'DmpHKDSatStatus.fPosition.fZ',
                        "DmpHKDSatStatus.fVelocity.fX", 
                        "DmpHKDSatStatus.fVelocity.fY", 
                        "DmpHKDSatStatus.fVelocity.fZ"])
    df = pd.DataFrame(data)
    df.columns = ['second', 'microsecond', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = df.sort_values(by=['second', 'microsecond']).drop_duplicates('second', keep='first')

    # UTC filtering to ensure no overlap between days
    utc_time = met_to_utc(df['second'], df['microsecond'])
    mask = (utc_time >= day_utc_start) & (utc_time < day_utc_end)
    df = df.loc[mask].reset_index(drop=True)
    df['utc'] = utc_time[mask]

    delta_t = (df['second'].to_numpy()[1:] - df['second'].to_numpy()[:-1]) \
            + 1e-6 * (df['microsecond'].to_numpy()[1:] - df['microsecond'].to_numpy()[:-1])
    delta_t = np.append(delta_t, 0.0)
    df['Delta_t'] = delta_t

    return df


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


def effective_area(cos_theta, phi, energy):
    cos_theta = np.asarray(cos_theta)
    energy_idx = np.digitize(energy, EAREA_E_BINS_EDGES) - 1
    costheta_idx = np.digitize(cos_theta, COSTHETA_BINS_EDGES) - 1
    phi_idx = np.digitize(phi, PHI_BINS_EDGES) - 1

    valid = (
        (energy_idx >= 0) & (energy_idx < EFF_AREA_TABLE.shape[0]) &
        (costheta_idx >= 0) & (costheta_idx < EFF_AREA_TABLE.shape[1]) &
        (phi_idx >= 0) & (phi_idx < EFF_AREA_TABLE.shape[2])
    )

    eff_area = np.zeros_like(cos_theta, dtype=float)

    # Only fill valid entries
    if valid.any():
        # Since energy_idx is scalar, just broadcast it:
        e_idx = energy_idx
        c_idx = costheta_idx[valid]
        p_idx = phi_idx[valid]

        eff_area[valid] = EFF_AREA_TABLE[e_idx, c_idx, p_idx]

    return eff_area

def transform_vectors_to_galactic(vectors):
    # Input: shape (N, 3)
    coords_icrs = SkyCoord(
        x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
        representation_type=CartesianRepresentation,
        frame='icrs'
    )
    coords_gal = coords_icrs.transform_to('galactic')
    return coords_gal.cartesian.xyz.value.T  # shape (N, 3)
    # Get only the 10 first entries of df
    df = df.head(10)
    lon_edges = np.linspace(-np.pi, np.pi, 720 + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    lat_centers = np.linspace(-np.pi/2, np.pi/2, 361)
    dlat = lat_centers[1] - lat_centers[0]
    lat_edges = np.zeros(len(lat_centers) + 1)
    lat_edges[0] = lat_centers[0] - dlat
    lat_edges[1:] = lat_centers

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')
    grid_coords = SkyCoord(lon_grid.ravel() * u.rad, lat_grid.ravel() * u.rad, frame='galactic')
    G = grid_coords.cartesian.xyz.value.T  # shape (M, 3)
    print(f"G created. Memory used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")


    def first_method(df):
        
        # Satellite positions and velocities in ICRS
        pos = np.vstack([df['x'].values, df['y'].values, df['z'].values]).T
        vel = np.vstack([df['vx'].values, df['vy'].values, df['vz'].values]).T
        times = df['utc'].values
        N = len(times)

        suz = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
        vel_proj = np.sum(vel * suz, axis=1, keepdims=True) * suz
        sux = vel - vel_proj
        sux /= np.linalg.norm(sux, axis=1, keepdims=True)
        suy = np.cross(suz, sux)

        exposure_map_flat = np.zeros(G.shape[0], dtype=np.float64)
        batch_size = 10
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            B = -suz[start_idx:end_idx]  # shape (B, 3) minus sign to get radial zenith
            BX = sux[start_idx:end_idx]
            BY = suy[start_idx:end_idx]

            B = transform_vectors_to_galactic(B)
            BX = transform_vectors_to_galactic(BX)
            BY = transform_vectors_to_galactic(BY)

            cos_theta = B @ G.T  # shape (B, M)
            print(f"cosTheta : {cos_theta}")
            # Reduce memory load
            cos_theta = cos_theta.astype(np.float32)
            print(f"cosTheta reduced. Memory used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

            # Compute phi in local spacecraft frame
            t_dot_sux = (BX @ G.T).astype(np.float32)
            t_dot_suy = (BY @ G.T).astype(np.float32)
            phi = np.arctan2(t_dot_suy, t_dot_sux).astype(np.float32)
           

            phi = np.mod(phi, 2 * np.pi)
            phi = np.degrees(phi)
            print(f"Phi : {phi}")

    def second_method(df):
        r = np.vstack([df['x'], df['y'], df['z']]).T
        r_hat = r / np.linalg.norm(r, axis=1)[:, None]
        obstime = met_to_utc(df['second'].to_numpy(), df['microsecond'].to_numpy())

        zenith_coords = SkyCoord(
            x=r_hat[:, 0], y=r_hat[:, 1], z=r_hat[:, 2],
            frame='icrs',
            representation_type=CartesianRepresentation
        ).transform_to('galactic')

        batch_size = 10
        N = len(zenith_coords)
        G = grid_coords.cartesian.xyz.value.T

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            B = zenith_coords.cartesian.xyz.value.T[start_idx:end_idx]
            cos_theta = B @ G.T
            print(f"cosTheta : {cos_theta}")

    def third_method(df):
        pos = np.vstack([df['x'].values, df['y'].values, df['z'].values]).T  # shape (N,3)
        vel = np.vstack([df['vx'].values, df['vy'].values, df['vz'].values]).T  # shape (N,3)
        N = pos.shape[0]
        # Satellite local frame axes
        suz = -pos / np.linalg.norm(pos, axis=1, keepdims=True)  # Nadir (Z)
        
        # Velocity component perpendicular to suz (forward direction, X)
        vel_proj = np.sum(vel * suz, axis=1, keepdims=True) * suz
        sux = vel - vel_proj
        sux /= np.linalg.norm(sux, axis=1, keepdims=True)
        
        # Y axis: suz × sux
        suy = np.cross(suz, sux)
        G_icrs = grid_coords.transform_to('icrs').cartesian.xyz.value.T  # shape (M, 3)
        G_icrs /= np.linalg.norm(G_icrs, axis=1, keepdims=True)
        M = G_icrs.shape[0]

        # Allocate output arrays
        cos_theta = np.empty((N, M), dtype=np.float32)
        phi_deg = np.empty((N, M), dtype=np.float32)


        BZ = suz
        BX = sux
        BY = suy

        cos_theta_batch = BZ @ G_icrs.T  # shape (B, M)
        t_dot_sux = BX @ G_icrs.T
        t_dot_suy = BY @ G_icrs.T

        phi_batch = np.arctan2(t_dot_suy, t_dot_sux)
        phi_batch = np.mod(phi_batch, 2 * np.pi)

        cos_theta = cos_theta_batch
        phi_deg = np.degrees(phi_batch)

        print(f"cosTheta : {cos_theta}"
              f"\nPhi : {phi_deg}")

        

    first_method(df)
    second_method(df)
    third_method(df)

def compute_exposure_map_roi(df, energy,
                             roi_center_lon, roi_center_lat, roi_side_deg=4, fov_angle_deg=60, verbose=False):

    """ Compute the exposure map for a given region of interest (ROI) in galactic coordinates.
    Args:
        df (pd.DataFrame): DataFrame containing satellite state data with columns ['x', 'y', 'z', 'vx', 'vy', 'vz', 'utc'].
        energy (float): Energy in GeV for which to compute the exposure.
        roi_center_lon (float): Center longitude of the ROI in degrees.
        roi_center_lat (float): Center latitude of the ROI in degrees.
        roi_side_deg (float): Side length of the square ROI in degrees.
        fov_angle_deg (float): Field of view angle in degrees.
    Returns:
        dict: A dictionary containing the exposure map and grid coordinates.
    """

    # Full resolution setup
    nlon, nlat = 2880, 1441
    lon_edges = np.linspace(-np.pi, np.pi, nlon + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    dlon = lon_edges[1] - lon_edges[0]

    lat_centers = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    dlat = lat_centers[1] - lat_centers[0]
    lat_edges = np.zeros(len(lat_centers) + 1)
    lat_edges[0] = lat_centers[0] - dlat
    lat_edges[1:] = lat_centers

    # ROI center in radians
    roi_center_lon_rad = np.deg2rad(roi_center_lon)
    roi_center_lat_rad = np.deg2rad(roi_center_lat)

    # Approx number of pixels to cover roi_side_deg
    n_pix_lon = int(np.round(np.deg2rad(roi_side_deg) / dlon))
    n_pix_lat = int(np.round(np.deg2rad(roi_side_deg) / dlat))

    # Force odd pixel counts
    if n_pix_lon % 2 == 0:
        n_pix_lon += 1
    if n_pix_lat % 2 == 0:
        n_pix_lat += 1

    half_pix_lon = n_pix_lon // 2
    half_pix_lat = n_pix_lat // 2

    # Find index of ROI center
    center_lon_idx = np.argmin(np.abs(lon_centers - roi_center_lon_rad))
    center_lat_idx = np.argmin(np.abs(lat_centers - roi_center_lat_rad))

    # Get indices symmetrically around center
    lon_indices = np.arange(center_lon_idx - half_pix_lon,
                            center_lon_idx + half_pix_lon + 1) % nlon  # wrap around
    lat_indices = np.arange(center_lat_idx - half_pix_lat,
                            center_lat_idx + half_pix_lat + 1)
    # Clip latitude to map bounds (no wrap-around at poles)
    lat_indices = np.clip(lat_indices, 0, nlat - 1)

    # Build ROI grid
    lon_grid_roi, lat_grid_roi = np.meshgrid(
        lon_centers[lon_indices], lat_centers[lat_indices], indexing='ij'
    )

    grid_coords = SkyCoord(lon_grid_roi.ravel() * u.rad,
                           lat_grid_roi.ravel() * u.rad,
                           frame='galactic')

    G_icrs = grid_coords.transform_to('icrs').cartesian.xyz.value.T  # shape (M, 3)
    print(f"G_icrs (ROI) created. Mem: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    # Satellite state
    pos = np.vstack([df['x'].values, df['y'].values, df['z'].values]).T
    vel = np.vstack([df['vx'].values, df['vy'].values, df['vz'].values]).T
    times = df['utc'].values
    N = len(times)

    suz = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
    vel_proj = np.sum(vel * suz, axis=1, keepdims=True) * suz
    sux = vel - vel_proj
    sux_norm = np.linalg.norm(sux, axis=1, keepdims=True)
    too_small = sux_norm[:, 0] < 1e-5
    fallback_axis = np.cross(np.array([[1, 0, 0]]), suz[too_small])
    sux[too_small] = fallback_axis
    sux /= np.linalg.norm(sux, axis=1, keepdims=True)
    suy = np.cross(suz, sux)
    sux = np.cross(suy, suz)  # Re-orthogonalize

    exposure_map_flat = np.zeros(G_icrs.shape[0], dtype=np.float64)
    batch_size = 1000
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        B = -suz[start_idx:end_idx]
        BX = sux[start_idx:end_idx]
        BY = suy[start_idx:end_idx]

        cos_theta = B @ G_icrs.T
        cos_theta = cos_theta.astype(np.float32)
        mask = cos_theta >= np.cos(np.radians(fov_angle_deg))
        if not np.any(mask):
            continue

        t_dot_sux = (BX @ G_icrs.T).astype(np.float32)
        t_dot_suy = (BY @ G_icrs.T).astype(np.float32)
        phi = np.rad2deg(np.mod(np.arctan2(t_dot_suy, t_dot_sux), 2 * np.pi))

        increments = np.zeros_like(cos_theta, dtype=np.float32)
        masked_theta = cos_theta[mask]
        masked_phi = phi[mask]
        increments[mask] = effective_area(masked_theta, masked_phi, energy)
        exposure_map_flat += increments.sum(axis=0)
        if verbose :
            print(f"Processed {start_idx}–{end_idx - 1}. Mem: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    exposure_map = exposure_map_flat.reshape(lon_grid_roi.shape)
    return {
        "exposure": exposure_map,
        "lon_indices" : lon_indices,
        "lat_indices" : lat_indices,
        "lon_grid": lon_grid_roi,
        "lat_grid": lat_grid_roi,
        "lon_centers": lon_centers[lon_indices],
        "lat_centers": lat_centers[lat_indices],
        "lon_edges": lon_edges[lon_indices[0]:lon_indices[-1] + 2],
        "lat_edges": lat_edges[lat_indices[0]:lat_indices[-1] + 2],
    }

def plot_exposure_map(lon_centers, lat_centers, exposure_map, target_lon, target_lat, target_name="Target", figPath=None):
    """ Plot the cropped exposure map in rectangular projection.
    Args:
        lon_centers (np.ndarray): 1D array of galactic longitudes in radians.
        lat_centers (np.ndarray): 1D array of galactic latitudes in radians.
        exposure_map (np.ndarray): 2D array of exposure values.
        figPath (str, optional): Path to save the figure. If None, the figure is shown.
    """

    # Set global font size
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

    # Convert radians to degrees for plotting
    lon_deg = np.rad2deg(lon_centers)
    lat_deg = np.rad2deg(lat_centers)

    # 2D grid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg, indexing='ij')

    # Transpose for imshow-like orientation
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(lon_grid, lat_grid, exposure_map, cmap='viridis', shading='auto')
    ax.plot(target_lon, target_lat, 'rx', markersize=10, label=f'{target_name}')
    ax.xaxis.set_inverted(True)  # Invert x-axis for galactic longitude
    ax.legend(loc='upper right')

    ax.set_xlabel('Galactic Longitude [deg]')
    ax.set_ylabel('Galactic Latitude [deg]')
    ax.set_title(f'Exposure Map of {target_name} ROI at {energy:.3f} GeV')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Exposure [cm² s sr]', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    ax.set_aspect('auto')
    ax.grid(True)

    fig.tight_layout()

    if figPath:
        plt.savefig(figPath)
        plt.close(fig)
    else:
        plt.show()

def process_day(date_str, energy, target, base_dir, result_dir, fig_dir, par_dir):
    """Process a single day's exposure map and save results."""

    day_start = Time(f"{date_str}T00:00:00", scale='utc')
    day_end = day_start + TimeDelta(1, format='jd')

    year, month, day = date_str.split("-")
    current_pattern = f"{base_dir}/{year}/{month}/{day}_data_photon.root"
    next_day_time = (day_start + TimeDelta(1, format='jd')).to_datetime(leap_second_strict='warn')
    next_pattern = f"{base_dir}/{next_day_time.year:04d}/{next_day_time.month:02d}/{next_day_time.day:02d}_data_photon.root"

    # Prepare output paths
    figPath_base = f"{fig_dir}/{target}/exposureMaps/{energy:.3f}GeV/{year}/{month}/"
    os.makedirs(figPath_base, exist_ok=True)
    figPath = f"{figPath_base}{date_str}.png"

    resultPath_base = f"{result_dir}/{energy:.3f}GeV/daily/"
    os.makedirs(resultPath_base, exist_ok=True)
    resultPath = f"{resultPath_base}{date_str}.npz"

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

    # Load data
    df = load_filtered_day_dataframe(current_pattern, next_pattern, day_start, day_end)
    if len(df) == 0:
        print(f"Warning: No data found for {date_str}")
        return

    # Apply SAA mask
    df, mask = mask_saa(df)

    # Compute exposure map
    res = compute_exposure_map_roi(df, energy, target_lon, target_lat)
    lon_edges = res['lon_edges']
    lat_edges = res['lat_edges']
    lon_centers = res['lon_centers']
    lat_centers = res['lat_centers']
    exposure_map = res['exposure']

    # Save outputs
    plot_exposure_map(lon_centers, lat_centers, exposure_map, target_lon, target_lat, target_name=target, figPath=figPath)
    np.savez(resultPath, lon_edges=lon_edges, lat_edges=lat_edges, exposure_map=exposure_map)
    print(f"Processed {date_str}: saved to {figPath}, {resultPath}")

    return res, target_lon, target_lat, target

def convert_longitude(lon_edges):
    """Convert longitudes from [0, 2π] to [-π, π] for Mollweide."""
    return -np.where(lon_edges > np.pi, lon_edges - 2 * np.pi, lon_edges)

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
    
    with np.load(exposure_files[0]) as ds:
        lon_edges = ds["lon_edges"]
        lat_edges = ds["lat_edges"]
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    for file in exposure_files:
        print(f"Loading {file}")
        data = np.load(file)
        exposure_map = data["exposure_map"]  # adjust key if needed

        if summed_map is None:
            summed_map = np.zeros_like(exposure_map)
        
        summed_map += exposure_map
        map_count += 1

    # Save the total summed map
    np.savez_compressed(output_path, exposure_map=summed_map)

    print(f"Summed {map_count} exposure maps.")
    print(f"Total summed map saved to {output_path}")

    return summed_map, lon_centers, lat_centers


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_exposure_map.py YEAR EnergyBinIndex Target")
        print("YEAR: either 'ALL' or a 4-digit year (e.g. 2018)")
        sys.exit(1)

    date_input = sys.argv[1]  # either "ALL" or a 4-digit year string
    energy_bin = int(sys.argv[2])
    target = sys.argv[3]

    energy = ENERGY_BINS_CENTERS[energy_bin] 

    base_dir = "/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2"
    result_dir = f"/dpnc/beegfs/dampe/tryf/ForKimberly/exposureMaps_targets/{target}/"
    fig_dir = "/home/users/l/lebrun/figures"
    par_dir = f"/home/users/l/lebrun/data/{target}"

    if date_input.lower() == "all":
        print("Processing all years from 2016 to 2024 sequentially.")

        years = [str(y) for y in range(2016, 2025)]  # exclude 2025
        all_dates = []

        for year in years:
            # Collect all dates for this year
            root_files = sorted(glob.glob(f"{base_dir}/{year}/[0-1][0-9]/*_data_photon.root", recursive=True))
            # Extract YYYY-MM-DD dates
            dates = sorted(set([
                f"{year}-" 
                f"{os.path.basename(os.path.dirname(f))}-"
                f"{os.path.basename(f).split('_')[0]}"
                for f in root_files
            ]))
            all_dates.extend(dates)

        dates = sorted(all_dates)
    else:
        # Expecting a year string like '2018'
        year = date_input
        if year == "2025":
            print("Year 2025 excluded from processing.")
            sys.exit(0)
        print(f"Processing year {year}.")

        # Gather all dates in this year
        root_files = sorted(glob.glob(f"{base_dir}/{year}/[0-1][0-9]/*_data_photon.root", recursive=True))
        dates = sorted(set([
            f"{year}-" 
            f"{os.path.basename(os.path.dirname(f))}-"
            f"{os.path.basename(f).split('_')[0]}"
            for f in root_files
        ]))

    if not dates:
        print(f"No data files found for input '{date_input}'. Exiting.")
        sys.exit(0)

    print(f"Processing dates from {dates[0]} to {dates[-1]} for target {target} at energy {energy:.3f} GeV.")

    for i, date_str in enumerate(dates):
        try:
            if i % 100 == 0:
                print(f"Processing day {i + 1}/{len(dates)}: {date_str}")
            res, target_lon, target_lat, target = process_day(date_str, energy, target, base_dir, result_dir, fig_dir, par_dir)
        except Exception as e:
            print(f"Error processing {date_str}: {e}")

    daily_result_dir = f"{result_dir}/{energy:.3f}GeV/daily/"
    output_path = f"{result_dir}/{energy:.3f}GeV/summed_exposure_map.npz"
    expoMap, lon_centers, lat_centers = sum_exposure_maps(daily_result_dir, output_path)
    print(f"Summed exposure map saved to {output_path}")

    # lon_centers = convert_longitude(lon_centers)

    plot_exposure_map(lon_centers, lat_centers, expoMap, target_lon, target_lat, target_name=target,
                      figPath=f"{fig_dir}/{target}/exposureMaps/{energy:.3f}GeV/summed_exposure_map.png")
