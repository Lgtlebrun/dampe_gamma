# %%
import numpy as np
import pandas as pd
import ROOT
import glob
import sys, os
import matplotlib.pyplot as plt

# %%

# Load the table once globally
PATH_EFFAREA = "/home/users/l/lebrun/results/effArea/effectiveArea_weighted.npz"
PATH_BINS = "/home/users/l/lebrun/results/effArea_bins_info_HET.npz"
# Load the effective area table
EFF_AREA_NPZ = np.load(PATH_EFFAREA)
EFF_AREA_TABLE = EFF_AREA_NPZ['effArea'] # Shape (n_E_bins, n_costheta_bins, n_phi_bins) ~ (20, 10, 360)

# Get bins edges and centers
ENERGY_BINS_EDGES = EFF_AREA_NPZ['E_bins_edge']
ENERGY_BINS_CENTERS = np.sqrt(ENERGY_BINS_EDGES[:-1] * ENERGY_BINS_EDGES[1:])  # GeV
COSTHETA_BINS_EDGES = EFF_AREA_NPZ['theta_bins_edge']
PHI_BINS_EDGES = EFF_AREA_NPZ['phi_bins_edge']


# %%
# Shape of effAre : [E, theta, phi] (20, 10, 36)
plt.figure(figsize=(8, 6))
phi_centers = 0.5 * (PHI_BINS_EDGES[:-1] + PHI_BINS_EDGES[1:])
costheta_centers = 0.5 * (COSTHETA_BINS_EDGES[:-1] + COSTHETA_BINS_EDGES[1:])
phi_grid, costheta_grid = np.meshgrid(phi_centers, costheta_centers, indexing='ij')

# Use bin edges for pcolormesh to match the shape of the data
plt.pcolormesh(PHI_BINS_EDGES, COSTHETA_BINS_EDGES, EFF_AREA_TABLE[5], shading='auto')
plt.xlabel('Phi [deg]')
plt.ylabel('cos(Theta)')
plt.title('Effective Area (Energy bin 5)')
plt.colorbar(label='Eff. Area')
plt.savefig("/home/users/l/lebrun/figures/effArea/effArea_E_bin_5.png", dpi=300)


