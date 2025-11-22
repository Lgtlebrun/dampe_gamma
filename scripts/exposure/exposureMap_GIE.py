import numpy as np
import pandas as pd
import ROOT
import glob
import sys, os
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from astropy.coordinates import CartesianRepresentation, SkyCoord, angular_separation, EarthLocation, GCRS, ITRS
from astropy.coordinates import Longitude, Latitude, AltAz
from astropy import units as u
from shapely.geometry import Point, Polygon
from shapely import vectorized
import psutil

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


def test_computation(df):
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

def compute_exposure_map(df, energy, res_n=2, fov_angle_deg=60):

    # Grid definition
    lon_edges = np.linspace(-np.pi, np.pi, 720 + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    dlon = lon_edges[1] - lon_edges[0]

    lat_centers = np.linspace(-np.pi/2, np.pi/2, 361)
    dlat = lat_centers[1] - lat_centers[0]
    lat_edges = np.zeros(len(lat_centers) + 1)
    lat_edges[0] = lat_centers[0] - dlat
    lat_edges[1:] = lat_centers

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')
    grid_coords = SkyCoord(lon_grid.ravel() * u.rad, lat_grid.ravel() * u.rad, frame='galactic')
    G = grid_coords.cartesian.xyz.value.T  # (M, 3)
    G_icrs = grid_coords.transform_to('icrs').cartesian.xyz.value.T  # shape (M, 3)
    print(f"G created. Memory used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    # Satellite state
    pos = np.vstack([df['x'].values, df['y'].values, df['z'].values]).T
    vel = np.vstack([df['vx'].values, df['vy'].values, df['vz'].values]).T
    times = df['utc'].values
    N = len(times)

    # Define Z axis: nadir (toward Earth center)
    suz = -pos / np.linalg.norm(pos, axis=1, keepdims=True)

    # Project vel into plane perpendicular to suz to define X axis
    vel_proj = np.sum(vel * suz, axis=1, keepdims=True) * suz
    sux = vel - vel_proj
    sux_norm = np.linalg.norm(sux, axis=1, keepdims=True)

    # Detect near-degenerate velocity projection (e.g. polar passes)
    too_small = sux_norm[:, 0] < 1e-5
    fallback_axis = np.cross(np.array([[1, 0, 0]]), suz[too_small])
    sux[too_small] = fallback_axis
    sux /= np.linalg.norm(sux, axis=1, keepdims=True)

    # Orthonormal frame
    suy = np.cross(suz, sux)
    sux = np.cross(suy, suz)  # Re-orthogonalize to avoid drift

    # Exposure map
    exposure_map_flat = np.zeros(G.shape[0], dtype=np.float64)
    print(f"exp map skeleton created. Memory used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    batch_size = 1000
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)

        B = -suz[start_idx:end_idx]  # Zenith direction (toward sky)
        BX = sux[start_idx:end_idx]
        BY = suy[start_idx:end_idx]

        cos_theta = B @ G_icrs.T  # (B, M)
        cos_theta = cos_theta.astype(np.float32)
        print(f"cosTheta created. Memory used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

        mask = cos_theta >= np.cos(np.radians(fov_angle_deg))
        if not np.any(mask):
            print(f"Skipping batch {start_idx}–{end_idx} (no points in FOV)")
            continue

        # Compute azimuth φ in spacecraft local frame
        t_dot_sux = (BX @ G_icrs.T).astype(np.float32)
        t_dot_suy = (BY @ G_icrs.T).astype(np.float32)
        phi = np.rad2deg(np.mod(np.arctan2(t_dot_suy, t_dot_sux), 2 * np.pi))  # Ensure [0, 2π) -> deg

        # Apply mask and compute effective area
        increments = np.zeros_like(cos_theta, dtype=np.float32)
        masked_theta = cos_theta[mask]
        masked_phi = phi[mask]
        increments[mask] = effective_area(masked_theta, masked_phi, energy)
        print(f"Incremented. Memory used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

        exposure_map_flat += increments.sum(axis=0)
        print(f"Processed timestamps {start_idx} to {end_idx - 1}")

    exposure_map = exposure_map_flat.reshape(lon_grid.shape)
    return exposure_map, lon_grid, lat_grid, lon_edges, lat_edges

def plot_exposure_map(lon_centers, lat_centers, exposure_map, figPath=None):
    """ Plot the exposure map in Mollweide projection.  
    Args:
        lon_centers (np.ndarray): 1D array of galactic longitudes in radians.
        lat_centers (np.ndarray): 1D array of galactic latitudes in radians.
        exposure_map (np.ndarray): 2D array of exposure values.
        figPath (str, optional): Path to save the figure. If None, the figure is shown.
    """

    # Convert your 1D radian lon/lat centers into 2D grid
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')

    # Convert to Mollweide's expected range: [-π, π] in lon, [-π/2, π/2] in lat
    lon_plot = lon_grid.T  # Transpose to match shape
    lat_plot = lat_grid.T
    exposure_plot = exposure_map.T  # Transpose as well to match pcolormesh layout

    # Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    # Plot with pcolormesh (lon, lat must be in radians)
    mesh = ax.pcolormesh(lon_plot, lat_plot, exposure_plot, cmap='viridis', shading='auto')

    # Final touches
    fig.colorbar(mesh, ax=ax, label='Exposure [cm² s sr]')
    ax.set_title('Exposure Map in Galactic Coordinates (Mollweide Projection)')
    tick_labels = np.array([150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150])
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Galactic Longitude')
    ax.set_ylabel('Galactic Latitude')
    ax.grid(True)

    if figPath:
        plt.savefig(figPath)
        plt.close(fig)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python compute_exposure_map.py YYYY-MM-DD Energy[GeV]")
        sys.exit(1)

    date_str = sys.argv[1]
    energy_bin = sys.argv[2]
    energy = ENERGY_BINS_CENTERS[int(energy_bin)]  # Convert energy bin to GeV
    day_start = Time(f"{date_str}T00:00:00", scale='utc')
    day_end = day_start + TimeDelta(1, format='jd')

    # File pattern for current and next day
    year, month, day = date_str.split("-")
    base = "/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2"
    current_pattern = f"{base}/{year}/{month}/{day}_data_photon.root"
    next_day_time = (day_start + TimeDelta(1, format='jd')).to_datetime(leap_second_strict='warn')
    next_pattern = f"{base}/{next_day_time.year:04d}/{next_day_time.month:02d}/{next_day_time.day:02d}_data_photon.root"

    figPath_base = f"/home/users/l/lebrun/figures/exposureMaps_GIE/{energy:.3f}GeV/{year}/{month}/"

    try :
        if not os.path.exists(figPath_base):
            os.makedirs(figPath_base)
    except FileExistsError as e:
        pass

    figPath = figPath_base + f"{date_str}.png"

    resultPath_base = f"/dpnc/beegfs/dampe/tryf/ForKimberly/exposureMaps_GIE/{energy:.3f}GeV/daily/"

    try : 
        if not os.path.exists(resultPath_base):
            os.makedirs(resultPath_base)
    except FileExistsError as e:
        pass

    resultPath = resultPath_base + f"{date_str}.npz"


    df = load_filtered_day_dataframe(current_pattern, next_pattern, day_start, day_end)
    print(f"Loaded {len(df)} timestamps for {date_str}.")

    # Print start and end timestamps
    start_time = met_to_utc(df['second'].iloc[0], df['microsecond'].iloc[0])
    end_time = met_to_utc(df['second'].iloc[-1], df['microsecond'].iloc[-1])
    print(f"Start time: {start_time.iso}, End time: {end_time.iso}")

    df, mask = mask_saa(df)

    test_computation(df)

    print(f"Masked out {np.sum(~mask)} timestamps inside SAA.")
    exposure_map, lon_grid, lat_grid, lon_edges, lat_edges = compute_exposure_map(df, energy)
    plot_exposure_map(lon_edges, lat_edges, exposure_map, figPath=figPath)
    print(f"Exposure map computed for {date_str} and saved to {figPath}.")

    # Save to disk
    np.savez(resultPath, lon_edges=lon_edges, lat_edges=lat_edges, exposure_map=exposure_map)
    
