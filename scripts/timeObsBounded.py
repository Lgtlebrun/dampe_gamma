from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ROOT, glob, sys, os, datetime
from astropy.time import Time, TimeDelta

from shapely.geometry import Point, Polygon
from shapely import vectorized

from astropy.coordinates import CartesianRepresentation, SkyCoord, GCRS, ITRS
import astropy.units as u
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


plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

treename='HousekeepingData/SatStatus'

cos_theta_min = 0.5
cos_theta_max = 1
n_theta_bins = 10

phi_min = 0
phi_max = 360
n_phi_bins = 360

theta_bins_edge = np.linspace(cos_theta_min, cos_theta_max, n_theta_bins + 1)
theta_bins_center = (theta_bins_edge[:-1] + theta_bins_edge[1:])/2

phi_bins_edge = np.linspace(phi_min, phi_max, n_phi_bins + 1)
phi_bins_center = (phi_bins_edge[:-1] + phi_bins_edge[1:])/2


def met_to_utc(seconds, microseconds):
    seconds = np.asarray(seconds)
    microseconds = np.asarray(microseconds)
    delta_sec = TimeDelta(seconds, format='sec')
    delta_us = TimeDelta(microseconds * 1e-6, format='sec')
    epoch = Time("2013-01-01T00:00:00", scale="utc")
    return epoch + delta_sec + delta_us

def met_to_utc_string(seconds, microseconds):
    # Ensure input is numpy array
    full_time = met_to_utc(seconds, microseconds)
    # Return human-readable UTC string
    return full_time.utc.iso

def load_day_df(file, day_start_str, verbose=True):

    day_start = Time(f"{day_start_str}T00:00:00", scale='utc')
    day_end = day_start + TimeDelta(1, format='jd') 

    rdf = ROOT.RDataFrame(treename, file)
    if verbose : print("Created RDataFrame")
    df = rdf.AsNumpy(['DmpHKDSatStatus.fSecond', 'DmpHKDSatStatus.fMicrosecond',
                    'DmpHKDSatStatus.fPosition.fX', 
                    'DmpHKDSatStatus.fPosition.fY', 
                    'DmpHKDSatStatus.fPosition.fZ',
                    "DmpHKDSatStatus.fVelocity.fX", 
                    "DmpHKDSatStatus.fVelocity.fY", 
                    "DmpHKDSatStatus.fVelocity.fZ"])
    if verbose : print("Converted to numpy")
    df = pd.DataFrame(df)

    df.rename(columns={'DmpHKDSatStatus.fSecond': 'second', 
                    'DmpHKDSatStatus.fMicrosecond': 'microsecond',
                    'DmpHKDSatStatus.fPosition.fX': 'x',
                    'DmpHKDSatStatus.fPosition.fY': 'y',
                    'DmpHKDSatStatus.fPosition.fZ': 'z',
                    'DmpHKDSatStatus.fVelocity.fX' : 'vx', 
                    'DmpHKDSatStatus.fVelocity.fY' : 'vy', 
                    'DmpHKDSatStatus.fVelocity.fZ' : 'vz', 
                    'DmpHKDSatStatus.fAngle' : 'fAngle'}, inplace=True)
    if verbose : print("Converted to pandas")
    # Print the number of entries
    if verbose : print("Number of entries:", len(df['second']))
    df = df.sort_values(by=['second', 'microsecond']).drop_duplicates('second', keep='first')
    # UTC filtering to ensure no overlap between days
    utc_time = met_to_utc(df['second'], df['microsecond'])
    mask = (utc_time >= day_start) & (utc_time < day_end)
    df = df.loc[mask].reset_index(drop=True)
    df['utc'] = utc_time[mask]
    # Print the number of entries with Delta_t != NaN
    if verbose : print("Number of entries w/o duplicates:", len(df['second']))
    if verbose : print(f"Timestamps span from {met_to_utc_string(df.iloc[0]['second'], df.iloc[0]['microsecond'])} to {met_to_utc_string(df.iloc[-1]['second'], df.iloc[-1]['microsecond'])}")

    return df
        
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

import re

def extract_date(path):
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})_', path)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None


def get_cos_theta_phi(df, target_gal):
    # Satellite positions and velocities
    pos = np.vstack([df['x'].values, df['y'].values, df['z'].values]).T  # shape (N,3)
    vel = np.vstack([df['vx'].values, df['vy'].values, df['vz'].values]).T  # shape (N,3)
    
    # Satellite local frame axes
    suz = -pos / np.linalg.norm(pos, axis=1, keepdims=True)  # Nadir (Z)
    
    # Velocity component perpendicular to suz (forward direction, X)
    vel_proj = np.sum(vel * suz, axis=1, keepdims=True) * suz
    sux = vel - vel_proj
    sux /= np.linalg.norm(sux, axis=1, keepdims=True)
    
    # Y axis: suz × sux
    suy = np.cross(suz, sux)
    
    # Target vector in GCRS frame
    obstime = df['utc'].values
    target_gcrs = target_gal.transform_to(GCRS(obstime=obstime))
    t_vec = target_gcrs.cartesian.get_xyz().value.T
    t_unit = t_vec / np.linalg.norm(t_vec, axis=1, keepdims=True)

    # Compute cos(theta) = nadir · target
    cos_theta = np.sum(t_unit * suz, axis=1)

    # Project target into sux-suy plane
    t_dot_sux = np.sum(t_unit * sux, axis=1)
    t_dot_suy = np.sum(t_unit * suy, axis=1)
    
    # Azimuth (phi) in satellite local frame
    phi = np.arctan2(t_dot_suy, t_dot_sux)  # in radians, in satellite frame
    phi = np.mod(phi, 2 * np.pi)  # wrap to [0, 2pi)
    
    return cos_theta, np.degrees(phi)  # return azimuth in degrees

def plot_histogram(histogram, figName):    
    plt.figure(figsize=(8, 5))
    plt.step(theta_bins_center, histogram, where='mid', color='blue', label='observing time')
    plt.xlabel(r'$\cos(\theta)$')
    plt.ylabel('Observing Time [s]')
    plt.title('Observation Distribution vs. $\cos(\\theta)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figName)

def plot_histogram2d(histogram2d, figName, start_date:datetime.datetime=None, end_date:datetime.datetime=None):
    plt.figure(figsize=(8, 6))

    # Meshgrid of bin edges (assumes you have global theta_bins_edge and phi_bins_edge)
    phi_grid, theta_grid = np.meshgrid(phi_bins_edge, theta_bins_edge)

    # Plot as 2D colormap
    pcm = plt.pcolormesh(phi_grid, theta_grid, histogram2d, shading='auto', cmap='viridis')

    title ='2D Observation Distribution'
    if start_date and end_date :
        str_start = start_date.strftime("%Y-%m-%d")
        str_end = end_date.strftime('%Y-%m-%d')
        title += f' ({str_start} to {str_end})'

    plt.xlabel(r'$\phi$ [deg]')
    plt.ylabel(r'$\cos(\theta)$')   
    plt.title(title)
    cbar = plt.colorbar(pcm, label='Observing Time [s]')
    plt.tight_layout()
    plt.savefig(figName)

def save_histo(histogram, fileName):
    np.savez(fileName, histogram=histogram)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python timeObs.py l0[deg] b0[deg] start_date[YYYY-MM-DD] end_date[YYYY-MM-DD] name")
        sys.exit(1)

    target_l = float(sys.argv[1])
    target_b = float(sys.argv[2])
    start_date = datetime.datetime.strptime(sys.argv[3], "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(sys.argv[4], "%Y-%m-%d").date()
    target_name = sys.argv[5]

    print(f"Computing observation time for target '{target_name}'")
    print(f"Target galactic coordinates: l={target_l}°, b={target_b}°")
    print(f"Date range: {start_date} → {end_date}")

    target_gal = SkyCoord(l=target_l, b=target_b, unit='deg', frame='galactic')

    # Paths
    dirDir = "/home/users/l/lebrun/"
    resultPathRoot = dirDir + "results/obsTime/"
    figPathRoot = dirDir + "figures/obsTime/"
    dataRootPath = "/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2/"

    histogram_total = np.zeros((len(theta_bins_edge) - 1, len(phi_bins_edge) - 1))

    # Loop over all months in date range
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        month_str = f"{month:02d}"

        dataPath = f"{dataRootPath}{year}/{month_str}/"
        print(f"\n[INFO] Searching data in: {dataPath}")

        files = sorted(glob.glob(dataPath + "*_data_photon.root"))
        if not files:
            print(f"[WARNING] No files found in {dataPath}")
        else:
            print(f"[INFO] Found {len(files)} files")

        for file in files:
            file_date_str = extract_date(file)
            file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d").date()

            if not (start_date <= file_date <= end_date):
                print(f"[DEBUG] Skipping {file} (date {file_date} out of range)")
                continue

            print(f"[INFO] Processing file: {file} (date {file_date})")
            df = load_day_df(file, file_date, verbose=False)
            df, valid_mask = mask_saa(df)

            cos_theta, phi = get_cos_theta_phi(df, target_gal)

            daily_hist, _, _ = np.histogram2d(cos_theta, phi, bins=[theta_bins_edge, phi_bins_edge])
            histogram_total += daily_hist

        current_date = (current_date.replace(day=1) + relativedelta(months=1))

    if not histogram_total.any():
        print("[ERROR] No data accumulated over the date range. Exiting.")
        sys.exit(1)

    # Create output directories
    os.makedirs(resultPathRoot, exist_ok=True)
    os.makedirs(figPathRoot, exist_ok=True)

    # Save total histogram
    fileName2D = f"{resultPathRoot}{target_name}_{start_date}_{end_date}_histogram.npz"
    figName2D = f"{figPathRoot}{target_name}_{start_date}_{end_date}_histogram2D.png"

    save_histo(histogram_total, fileName2D)
    plot_histogram2d(histogram_total, figName2D, start_date, end_date)

    print(f"\n✅ Total histogram saved to {fileName2D}")
    print(f"✅ Total figure saved to {figName2D}")

