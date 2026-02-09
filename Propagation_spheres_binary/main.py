from grating import *
from sample import *
from test_samples import *
from propagator import *
import numpy as np
from parameters import *
from detector import *
from plotting import *
import time
import threading
import os
import csv
import matplotlib.pyplot as plt

samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                d_sph_in_um = d_sph_in_um,
                f_sph = f_sph,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg,
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 
"""
samp2d = test_Sample(t_samp_in_mm = t_samp_in_mm,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg, 
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 
"""
grat1d = Grating(px_in_um = px_in_um)

"""prop = Propagator(grat = grat1d,
                        samp = samp2d,
                        prop_in_m = prop_in_m)"""

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
        data = np.load(slice_profiles_path)
        slc2d_sph_padded = data['slc2d_sph_padded']
        slc2d_bkg_padded = data['slc2d_bkg_padded'] 
        print()
        print("Loaded slice profiles from file.")
    else:
        # Create and save
        slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices(seed=0)
        np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
        print()
        print("Created and saved slice profiles.")

    #plot_intensity_withG2(det, prop,  wavefld_bg, save_plot=True)
    #plot_intensity_withoutG2(det, prop,  wavefld_bg, save_plot=True)
    #print('visibility',calc_vis_with_G2(det))

    def size_change(det: Detector,
                                prop: Propagator,
                                wavefld_bg: np.ndarray,
                                bin_grat: np.ndarray) -> tuple:
        results = []
        sphere_sizes_in_um = np.arange(2, 31)
        for sphere_size in sphere_sizes_in_um:
            samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                        d_sph_in_um = sphere_size,
                        f_sph = f_sph,
                        mat_sph_type = mat_sph_type,
                        mat_bkg_type = mat_bkg_type,
                        mat_sph = mat_sph, 
                        mat_bkg = mat_bkg,
                        rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                        rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 
            """
            prop = Propagator(grat = grat1d,
                            samp = samp2d,
                            prop_in_m = prop_in_m)
            """
            slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices(seed=0)
            np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
            visibility, epsilon = save_visibility_epsilon(det, prop,  wavefld_bg, prop.bin_grat, thick_samp_mm=t_samp_in_mm)
            print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV at diameter: {sphere_size:.2f}")
            particle_fraction = np.sum(slc2d_sph_padded)/(samp_size_in_pix*t_samp_in_mm * 1e-3/sim_pix_size_in_m)
            results.append([sphere_size, visibility.real, epsilon.real, particle_fraction, samp2d.num_sph_2dslice ])
            os.remove(slice_profiles_path)
            print(particle_fraction, "real particle fraction")
            del slc2d_sph_padded, slc2d_bkg_padded

            with open("visibility_results_10keV_1mm_slc=2.8um_v2.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Sphere size (um)", "Visibility", "Epsilon", "Particle fraction", "Number of spheres"])
                writer.writerows(results)

    results = []
    #distances_samp_det = np.arange(0.01, 0.39, 0.02)   # in m
    distances = np.arange(0, 0.91, 0.02)
    for distance in distances:

        prop = Propagator(grat = grat1d,
                    samp = samp2d,
                    prop_in_m = distance)
        #visibility, epsilon = save_visibility_epsilon(det, prop,  wavefld_bg, prop.bin_grat, t_samp_in_mm)
        visibility, epsilon = save_visibility_epsilon_sections(det, prop,  wavefld_bg, prop.bin_grat, t_samp_in_mm)
        corr_length = (l_in_m*distance)/(px_in_um * 1e-6) 
        print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV distance to detector: {distance*1e2:.2f} cm")
        print(f"Correlation length: {corr_length*1e6:.2f} um")
        particle_fraction = np.sum(slc2d_sph_padded)/(samp_size_in_pix*t_samp_in_mm * 1e-3/sim_pix_size_in_m)
        results.append([distance, corr_length, visibility.real, epsilon.real])
        print(particle_fraction, "real particle fraction")
        #plot_intensity_withG2(det, prop,  wavefld_bg, save_plot=True)

        with open("Prade_remake_10um_results_45keV_corr_lengths_0966diam_sections.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Propagation distance","Correlation length", "Visibility", "Epsilon"])
            writer.writerows(results)



finally:

    end = time.time() 
    stop_event.set()
    timer_thread.join()
    print(f"Total simulation time: {end - start:.2f} seconds.")
    print()



# --- Save all the simulation parameters --------------------------------------

sim_param = {   
                "Energy in keV": E_in_keV, 
                "Simulated pixel size in m": sim_pix_size_in_m, 
                "Simulated image size in pix": img_size_in_pix,
                "Grating period in X in um": grat1d.px_in_um,                    
                "Sample size in pix": samp_size_in_pix,
                "Sample thickness in mm": samp2d.thickness_in_mm,
                #"Sphere diameter in um": samp2d.d_sph_in_um,
                #"Packing fraction of the spheres": samp2d.f_sph, 
                "Sphere material": samp2d.mat_sph,
                "Background material": samp2d.mat_bkg,
                #"Number of spheres": samp2d.num_sph_2dslice,
                "Number of slices": samp2d.num_slc,
                "Thickness of a sample slice in pix": t_slc_in_pix,
                "Talbot distance in m": round(prop.talbot_in_m, 1),
                "Grating-to-detector distance in cm": round(prop.grat2det_in_m * 100, 1),
                "Grating-to-sample distance in cm": round(prop.grat2samp_in_m * 100, 1),
                "Sample-to-detector distance in cm": round(prop.samp2det_in_m * 100, 1),
                #"Propagation distance in cm": round(prop_in_m * 100, 1), 
                "Simulation time in min": round((end - start) / 60, 1)
            }

with open("sim_param.csv",
          "w", 
          newline = "") as file:
    w = csv.writer(file)
    w.writerows(sim_param.items())
