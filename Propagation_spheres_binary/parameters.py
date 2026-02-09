import numpy as np
from scipy import constants as cte
import sys
from pathlib import Path
import xraydb as xdb

# --- Grating -----------------------------------------------------------------

#grating period of G1 in um
px_in_um = 10

# --- Geometry ----------------------------------------------------------------

# Simulated pixel size in m
sim_pix_size_in_m = 1e-8

# Image size in pix, decides the length in x direction of the setup
# you can either give it a fixed amount of grating periods or just straight up decide the amound of pixels
# I recommend an even number (even better is power 2) of grating periods because of Fourier effects
img_size_in_pix = 128 * int(round(px_in_um * 1e-6 / sim_pix_size_in_m))

# Sample size in pix in x direction (as I don't really do padding I just set them the same)
samp_size_in_pix = img_size_in_pix

# Sample size in meters in x direction
samp_size_in_m = samp_size_in_pix * sim_pix_size_in_m

# Distance from middle of sample to detector
prop_in_m = 0.175

# --- Source ------------------------------------------------------------------

# Energy in keV
E_in_keV = 45
E_in_J = E_in_keV * 1e3 * cte.e

# Number of photons per pixel
num_ph = 1e5

# Wavelength in m 
l_in_m = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_in_keV * 1e3)  

# Wavevector magnitude in 1/m 
k_in_1_m = 2 * np.pi * (E_in_keV * 1e3) / \
           (cte.physical_constants["Planck constant in eV s"][0] * cte.c)

# Electron radius
r_e = cte.physical_constants["classical electron radius"][0]  # in m

#correlation length in meters
xi_corr = prop_in_m * l_in_m / (px_in_um*1e-6)

# --- Sample ------------------------------------------------------------------

# total sample thickness in mm
t_samp_in_mm = 1

# sohere diameter in um
d_sph_in_um = 0.966

#particle fraction of spheres in the sample
f_sph = 0.102

# thickness of the individual slices in um
t_slc_in_um = 1.4

# thickness of the individual slices in pixels
t_slc_in_pix = int(t_slc_in_um * 1e-6 / sim_pix_size_in_m)

mat_sph_type = "compound"
mat_bkg_type = "compound"

# Here just comment or uncomment the materials you would like to have for spheres and background

mat_sph = "SiO2"
name_mat_sph = "glass"
rho_sph_in_g_cm3 = 2.196

"""
mat_bkg = "C2H6O"
name_mat_bkg = "Ethanol"
rho_bkg_in_g_cm3 = 0.78945
"""

mat_bkg = "H2O"
name_mat_bkg = "Water"
rho_bkg_in_g_cm3 = 0.998

"""
name_sph = "Air"
mat_sph = "N0.78084O0.20946Ar0.00934C0.00036Ne0.000018He0.000005Kr0.000001" 
rho_sph_in_g_cm3 = 0.001225 

name_bkg = "bone"
mat_bkg ="H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529" 
rho_bkg_in_g_cm3 = 1.92 
"""

# --- Detector ----------------------------------------------------------------

# size of bins in case we do any binning
detector_pixel_size = 1*1e-6
binning_factor = int(detector_pixel_size/sim_pix_size_in_m)

# When we do reconstruction of the darkfield image and look at the intensity in sections we decide 
# here how big those sections are (I recommend to have it set to the grating period as be get a min 
# and a max in each section)
segment_size_in_um = px_in_um
segment_size_in_pix = int(round(segment_size_in_um * 1e-6 / sim_pix_size_in_m))
