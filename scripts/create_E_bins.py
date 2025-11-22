import numpy as np

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


if __name__ == "__main__":

    dirDir= "/home/users/l/lebrun/"
    resultPath = dirDir + "results/"

    energy_centers_GIEM = np.array([1.16121705e+00, 1.50918340e+00, 1.96142017e+00, 2.54917267e+00,
 3.31304908e+00, 4.30582611e+00, 7.27300221e+00, 1.22848810e+01,
 2.07504819e+01, 3.50497899e+01, 5.92028552e+01, 9.99999737e+01,
 1.68910683e+02, 2.85308264e+02, 4.81916266e+02, 8.14008272e+02])
    ratios = energy_centers_GIEM[1:] / energy_centers_GIEM[:-1]
    coarse_ratio = ratios[-1]

    # Add 4 more bins following the coarse spacing
    extended_centers = list(np.sort(energy_centers_GIEM))
    last_center = energy_centers_GIEM[-1]

    for i in range(4):
        next_center = last_center * coarse_ratio
        extended_centers.append(next_center)
        last_center = next_center

    extended_centers = np.array(extended_centers)

    # Calculate edges for the extended centers
    E_bins_edge_HET = calculate_bin_edges(extended_centers)

    # E_bins_edge_HET = np.logspace(np.log10(Emin), np.log10(Emax_HET), n_E_bins_HET + 1)
    E_bins_center_HET = np.sqrt(E_bins_edge_HET[:-1] * E_bins_edge_HET[1:])

    n_E_bins_HET = len(E_bins_center_HET)

    new_ratios = E_bins_center_HET[1:] / E_bins_center_HET[:-1]

    print(f"Computed bin centers : {E_bins_center_HET}")
    print(f"Computed bin edges : {E_bins_edge_HET}")

    print(f"Computed ratios : {new_ratios}")


    cos_theta_min = 0.5
    cos_theta_max = 1
    phi_min = 0
    phi_max = 360
    n_theta_bins = 10
    n_phi_bins = 360


    theta_bins_edge = np.linspace(cos_theta_min, cos_theta_max, n_theta_bins + 1)
    theta_bins_center = (theta_bins_edge[:-1] + theta_bins_edge[1:])/2

    phi_bins_edge = np.linspace(phi_min, phi_max, n_phi_bins + 1)
    phi_bins_center = (phi_bins_edge[:-1] + phi_bins_edge[1:])/2

    lon_edges = np.linspace(-np.pi, np.pi, 2880 + 1)

    lat_centers = np.linspace(-np.pi/2, np.pi/2, 1441)
    dlat = lat_centers[1] - lat_centers[0]
    lat_edges = np.zeros(len(lat_centers) + 1)
    lat_edges[0] = lat_centers[0] - dlat
    lat_edges[1:] = lat_centers

    print(f"Computed theta edges : {theta_bins_edge}")
    print(f"Computed phi edges : {phi_bins_edge}")
    print(f"Computed lon edges : {lon_edges}")
    print(f"Computed lat edges : {lat_edges}")


    np.savez(resultPath + "all_bins_GIEM.npz", E_bins_edge=E_bins_edge_HET, theta_bins_edge=theta_bins_edge,
             phi_bins_edge=phi_bins_edge, lon_edges=lon_edges, lat_edges=lat_edges)