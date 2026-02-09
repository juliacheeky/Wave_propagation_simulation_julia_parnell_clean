from skull_grating import *
from skull_propagator import *
import numpy as np
from skull_parameter import *
from skull_detector import *
from skull_plotting import *
from skull_sample import *
import time
import threading
import os
import csv
import cv2  
import matplotlib.pyplot as plt
import math

sampskull = Sample_Skull(mat_bone = mat_bone, mat_air = mat_air) 

grat1d = Grating(px_in_um = px_in_um)

"""prop = Propagator(grat = grat1d,
                    samp = sampskull,
                    prop_in_m = prop_in_m,
                    mat_bone = mat_bone,
                    mat_air = mat_air,
                    energy = E_in_keV,
                    rho_bone_in_g_cm3 = rho_bone_in_g_cm3,
                    rho_air_in_g_cm3 = rho_air_in_g_cm3)"""

det = Detector(px_in_um= px_in_um)

wavefld_bg = np.ones(img_size_in_pix)

slice_profiles_path =  "slices_data.npz"

def print_elapsed_time(start_time, stop_event):
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        print(f"\rElapsed time: {elapsed:.1f} seconds", end="")
        time.sleep(1)
    print()  # Move to next line after stopping

start = time.time()
stop_event = threading.Event()
timer_thread = threading.Thread(target=print_elapsed_time, args=(start, stop_event))
timer_thread.start()

try:
    
    if os.path.exists(slice_profiles_path):
        # Load from file
        print()
        print("Loaded slice profiles from file.")
    else:
        # Create and save
        #positive, negative = sampskull.create_slice2d()
        bone, pores = sampskull.create_projected_1d_slices()
        #np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
        np.savez(slice_profiles_path, slc2d_sph_padded=bone, slc2d_bkg_padded=pores)
        print()
        print("Created and saved slice profiles.")
    """
    prop = Propagator(grat = grat1d,
                    samp = sampskull,
                    prop_in_m = 0.4,
                    mat_bone = mat_bone,
                    mat_air = mat_air,
                    energy = E_in_keV,
                    rho_bone_in_g_cm3 = rho_bone_in_g_cm3,
                    rho_air_in_g_cm3 = rho_air_in_g_cm3)
    plot_intensity_withG2(E_in_keV,det, prop,  wavefld_bg, save_plot=True)
    """    
    results = []
    distances = np.arange(0, 2.41, 0.02)
    for distance in distances:

        prop = Propagator(grat = grat1d,
                    samp = sampskull,
                    prop_in_m = distance,
                    mat_bone = mat_bone,
                    mat_air = mat_air,
                    energy = E_in_keV,
                    rho_bone_in_g_cm3 = rho_bone_in_g_cm3,
                    rho_air_in_g_cm3 = rho_air_in_g_cm3)
        #visibility, epsilon , mean_transmission = save_visibility_epsilon(det, prop,  wavefld_bg, prop.bin_grat,thick_samp_mm=sampskull.thickness_in_mm)
        visibility, epsilon = save_visibility_epsilon_sections(det, prop,  wavefld_bg, prop.bin_grat, thick_samp_mm=sampskull.thickness_in_mm)
        corr_length = (l_in_m*distance)/(px_in_um * 1e-6) 
        print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV distance to detector: {distance*1e2:.2f} cm")
        print(f"Correlation length: {corr_length*1e6:.2f} um")
        results.append([distance, corr_length, visibility.real, epsilon.real])

        with open("04-02__rec_Tra0548_simulated_60keV_corr_segs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Prop distance","Correlation Length","Visibility","Epsilon"])
            writer.writerows(results)
finally:

    end = time.time() 
    stop_event.set()
    timer_thread.join()
    print(f"Total simulation time: {end - start:.2f} seconds.")
    print()




    """    energies = list(range(10, 101, 5))
    for energy in energies:
        prop = Propagator(grat = grat1d,
                    samp = sampskull,
                    prop_in_m = prop_in_m,
                    mat_bone = mat_bone,
                    mat_air = mat_air,
                    energy = energy,
                    rho_bone_in_g_cm3 = rho_bone_in_g_cm3,
                    rho_air_in_g_cm3 = rho_air_in_g_cm3)

        visibility, epsilon = fourier_fit_vis(prop,  wavefld_bg, grat1d,thick_samp_mm=sampskull.thickness_in_mm)
        #particle_fraction = np.sum(prop.slc2d_sph_full)/(samp_size_in_pix*sampskull.thickness_in_mm * 1e-3/sim_pix_size_in_m)
        print(f"Visibility with sample: {visibility.real:.3f} at Energy: {energy:.1f} keV ")
        results.append([energy, visibility.real, epsilon.real])
        with open("outer_10-7_Tra2224_v2-fourier_vis.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Energy", "Visibility", "Epsilon"])
            writer.writerows(results)"""
    


#Change in Energy and grating period simultaneously
"""   
    #px_um = np.arange(4, 14, 1)
    #px_um = list(range(4, 14, 1))

    values = 1/(np.sqrt(np.linspace(0.1, 0.01, 30)))
    px_um = np.round(values, 2).tolist()
    for period in px_um:
        lam = (period*1e-6)**2 / (2 * distance)
        E = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / lam * 1e-3
        print(f"Simulating for period: {period} um, Energy: {E} keV")
        grat1d = Grating(px_in_um = period)
        det = Detector(px_in_um= period)
        prop = Propagator(grat = grat1d,
                    samp = sampskull,
                    prop_in_m = prop_in_m,
                    mat_bone = mat_bone,
                    mat_air = mat_air,
                    energy = E,
                    rho_bone_in_g_cm3 = rho_bone_in_g_cm3,
                    rho_air_in_g_cm3 = rho_air_in_g_cm3)
        visibility, epsilon = save_visibility_epsilon(det, prop,  wavefld_bg, prop.bin_grat,thick_samp_mm=sampskull.thickness_in_mm)
        print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E:.1f} keV ")
        print(f"amount of grating periods: {img_size_in_pix/(period * 1e-6 / sim_pix_size_in_m)}")
        results.append([E, period,visibility.real, epsilon.real, img_size_in_pix/(period * 1e-6 / sim_pix_size_in_m)])
        """ 