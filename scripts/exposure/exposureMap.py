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

# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea_df_HET.csv"
PATH_BINS = "/home/users/l/lebrun/results/effArea_bins_info_HET.npz"
# Load the effective area table
EFF_AREA_BINS = np.load(PATH_BINS)
EFF_AREA_TABLE = pd.read_csv(PATH_EFFAREA)

# Get bins edges and centers
ENERGY_BINS_EDGES = EFF_AREA_BINS['E_bins_edge_HET']
ENERGY_BINS_CENTERS = EFF_AREA_BINS['E_bins_center_HET']
COSTHETA_BINS_EDGES = EFF_AREA_BINS['theta_bins_edge']
COSTHETA_BINS_CENTERS = EFF_AREA_BINS['theta_bins_center']

# Determine max bins sizes
max_energy_bin = EFF_AREA_TABLE['E_bin_HET'].max()
max_costheta_bin = EFF_AREA_TABLE['cos_theta_bin'].max()

# Create 2D lookup array initialized to 0
eff_area_array = np.zeros((max_energy_bin + 1, max_costheta_bin + 1))

# Fill the array with known values
for _, row in EFF_AREA_TABLE.iterrows():
    e_bin = int(row['E_bin_HET'])
    c_bin = int(row['cos_theta_bin'])
    eff_area_array[e_bin, c_bin] = row['effArea_w']

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
    - df: pandas.DataFrame with columns ['second', 'microsecond', 'x', 'y', 'z']
    
    Returns:
    - df_masked: DataFrame excluding SAA timestamps
    - mask: boolean mask array (True = keep, False = SAA)
    """
    # Convert MET to UTC Time objects
    obstime = met_to_utc(df['second'].to_numpy(), df['microsecond'].to_numpy())

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
    in_saa = (
        (lat >= -50) & (lat <= 0) &
        (lon >= -90) & (lon <= -30)
    )

    # Invert to create a mask of valid points
    valid_mask = ~in_saa
    df_masked = df[valid_mask].reset_index(drop=True)
    return df_masked, valid_mask

def load_filtered_day_dataframe(day_path, next_day_path, day_utc_start, day_utc_end):
    files = sorted(glob.glob(day_path) + glob.glob(next_day_path))
    rdf = ROOT.RDataFrame("HousekeepingData/SatStatus", files)
    data = rdf.AsNumpy(['DmpHKDSatStatus.fSecond', 'DmpHKDSatStatus.fMicrosecond',
                        'DmpHKDSatStatus.fPosition.fX', 'DmpHKDSatStatus.fPosition.fY', 'DmpHKDSatStatus.fPosition.fZ'])
    df = pd.DataFrame(data)
    df.columns = ['second', 'microsecond', 'x', 'y', 'z']
    df = df.sort_values(by=['second', 'microsecond']).drop_duplicates('second', keep='first')

    # UTC filtering to ensure no overlap between days
    utc_time = met_to_utc(df['second'], df['microsecond'])
    mask = (utc_time >= day_utc_start) & (utc_time < day_utc_end)
    df = df.loc[mask].reset_index(drop=True)

    delta_t = (df['second'].to_numpy()[1:] - df['second'].to_numpy()[:-1]) \
            + 1e-6 * (df['microsecond'].to_numpy()[1:] - df['microsecond'].to_numpy()[:-1])
    delta_t = np.append(delta_t, 0.0)
    df['Delta_t'] = delta_t

    return df

def effective_area(cos_theta, energy):
    cos_theta = np.asarray(cos_theta)
    energy_idx = np.digitize(energy, ENERGY_BINS_EDGES) - 1
    costheta_idx = np.digitize(cos_theta, COSTHETA_BINS_EDGES) - 1

    valid = (
        (energy_idx >= 0) & (energy_idx < eff_area_array.shape[0]) &
        (costheta_idx >= 0) & (costheta_idx < eff_area_array.shape[1])
    )

    eff_area = np.zeros_like(cos_theta, dtype=float)

    # Only fill valid entries
    if valid.any():
        # Since energy_idx is scalar, just broadcast it:
        e_idx = energy_idx
        c_idx = costheta_idx[valid]

        eff_area[valid] = eff_area_array[e_idx, c_idx]

    return eff_area

def compute_exposure_map(df, energy, res_n=2, fov_angle_deg=60):
    # UPDATE : hardcode to meet Fermi's requirement
    # lon_edges = np.linspace(-np.pi, np.pi, int(360 * res_n) + 1)
    # lat_edges = np.linspace(-np.pi/2, np.pi/2, int(180 * res_n) + 1)


    # lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    # lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    
    lon_edges = np.linspace(-np.pi, np.pi, 2880 + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    lat_centers = np.linspace(-np.pi/2, np.pi/2, 1441)
    dlat = lat_centers[1] - lat_centers[0]
    lat_edges = np.zeros(len(lat_centers) + 1)
    lat_edges[0] = lat_centers[0] - dlat
    lat_edges[1:] = lat_centers

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')
    grid_coords = SkyCoord(lon_grid.ravel() * u.rad, lat_grid.ravel() * u.rad, frame='galactic')

    r = np.vstack([df['x'], df['y'], df['z']]).T
    r_hat = r / np.linalg.norm(r, axis=1)[:, None]
    obstime = met_to_utc(df['second'].to_numpy(), df['microsecond'].to_numpy())

    zenith_coords = SkyCoord(
        x=r_hat[:, 0], y=r_hat[:, 1], z=r_hat[:, 2],
        frame='icrs',
        representation_type=CartesianRepresentation
    ).transform_to('galactic')

    batch_size = 1000
    N = len(zenith_coords)
    G = grid_coords.cartesian.xyz.value.T
    exposure_map_flat = np.zeros(G.shape[0], dtype=np.float64)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        B = zenith_coords.cartesian.xyz.value.T[start_idx:end_idx]
        cos_theta = B @ G.T
        mask = cos_theta >= np.cos(np.radians(fov_angle_deg))
        increments = np.zeros_like(cos_theta)
        increments[mask] = effective_area(cos_theta[mask], energy)
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
    fig.colorbar(mesh, ax=ax, label='Exposure')
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
    energy = sys.argv[2]
    day_start = Time(f"{date_str}T00:00:00", scale='utc')
    day_end = day_start + TimeDelta(1, format='jd')

    # File pattern for current and next day
    year, month, day = date_str.split("-")
    base = "/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2"
    current_pattern = f"{base}/{year}/{month}/{day}_data_photon.root"
    next_day_time = (day_start + TimeDelta(1, format='jd')).to_datetime()
    next_pattern = f"{base}/{next_day_time.year:04d}/{next_day_time.month:02d}/{next_day_time.day:02d}_data_photon.root"

    figPath_base = f"/home/users/l/lebrun/figures/exposureMaps/{energy}GeV/{year}/{month}/"

    try :
        if not os.path.exists(figPath_base):
            os.makedirs(figPath_base)
    except FileExistsError as e:
        pass

    figPath = figPath_base + f"{date_str}.png"

    resultPath_base = f"/dpnc/beegfs/dampe/tryf/ForKimberly/exposureMaps/{energy}GeV/"

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
    print(f"Masked out {np.sum(~mask)} timestamps inside SAA.")
    exposure_map, lon_grid, lat_grid, lon_edges, lat_edges = compute_exposure_map(df, energy)
    plot_exposure_map(lon_edges, lat_edges, exposure_map, figPath=figPath)
    print(f"Exposure map computed for {date_str} and saved to {figPath}.")

    # Save to disk
    np.savez(resultPath, lon_edges=lon_edges, lat_edges=lat_edges, exposure_map=exposure_map)
    
