import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from parameter import *
import xraylib as xrl

hight_of_pixel_in_um = 217
hight_of_pixel_in_pix = int(hight_of_pixel_in_um*1e-6 / detector_pixel_size)

additional_thickness_cm= 11.95
mu_bkg_in_1_cm = xrl.CS_Total_CP(mat_bkg, E_in_keV) \
                                 * rho_bkg_in_g_cm3

#print(np.exp(-mu_bkg_in_1_cm * additional_thickness_cm))
data  = "intensity_tumor_background_60keV_all_sizes_talbot3_large_sizes.csv"

"""
d_sph = 40
col_name = f"{int(round(d_sph))}um"

I_tumor = pd.read_csv(data)[col_name].to_numpy()
I_ref = pd.read_csv(data)["I_ref"].to_numpy()
I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()
"""
#I_tumor = I_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
#I_no_tumor = I_no_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
def estimate_phi(Iref, Isamp):
    omega = 2 * np.pi / (px_in_um*1e-6 / detector_pixel_size)
    x_walk = np.arange(len(Iref))
    
    centers = []
    phi_list = []

    for start in range(0, len(x_walk) - segment_size_in_pix + 1, segment_size_in_pix):
        
        x_seg = x_walk[start:start+segment_size_in_pix]
        Iref_seg = Iref[start:start+segment_size_in_pix]
        Isamp_seg = Isamp[start:start+segment_size_in_pix]

        # Linearized cosine model
        X = np.column_stack([
            np.ones_like(x_seg),
            np.cos(omega * x_seg),
            np.sin(omega * x_seg)
        ])

        beta_ref, _, _, _ = np.linalg.lstsq(X, Iref_seg, rcond=None)
        beta_samp, _, _, _ = np.linalg.lstsq(X, Isamp_seg, rcond=None)

        A_ref = beta_ref[1]
        A_samp = beta_samp[1]

        B_ref = beta_ref[2]
        B_samp = beta_samp[2]

        phi_ref = np.arctan2(-B_ref, A_ref) 
        phi_samp = np.arctan2(-B_samp, A_samp)

        centers.append(np.mean(x_seg))
        phi_list.append(phi_samp - phi_ref)

    return np.array(phi_list)


def compute_intensity_2D_pixels(I_ref, I_tumor, I_no_tumor, d_sph_um):
    d_sph_in_pix = int(d_sph_um*1e-6 / detector_pixel_size)
    I_ref_2D_pixel = hight_of_pixel_in_pix*I_ref
    print("sphere diameter in pix",d_sph_in_pix)
    I_tumor_2D_pixel = (hight_of_pixel_in_pix-d_sph_in_pix)*I_ref + d_sph_in_pix*I_tumor
    I_tumor_2D_pixel = I_tumor_2D_pixel / I_ref_2D_pixel
    I_no_tumor_2D_pixel = (hight_of_pixel_in_pix-d_sph_in_pix)*I_ref + d_sph_in_pix*I_no_tumor
    I_no_tumor_2D_pixel = I_no_tumor_2D_pixel / I_ref_2D_pixel
    return I_ref_2D_pixel, I_tumor_2D_pixel, I_no_tumor_2D_pixel

def estimate_phi_fourier(Iref, Isamp):
        omega = 2 * np.pi / (px_in_um*1e-6 / detector_pixel_size)
        x_walk = np.arange(len(Iref))
        
        centers = []
        phi_list = []
        mean_list =[]

        for start in range(0, len(x_walk) - segment_size_in_pix + 1, segment_size_in_pix):
            #print(f"Processing segment starting at pixel {start}")
            x_seg = x_walk[start:start+segment_size_in_pix]
            Iref_seg = Iref[start:start+segment_size_in_pix]
            Isamp_seg = Isamp[start:start+segment_size_in_pix]
            fourier_Isamp = np.fft.fft(Isamp_seg)
            fourier_Iref = np.fft.fft(Iref_seg)
            N = len(Isamp_seg)
            k = int(N / (px_in_um*1e-6 / detector_pixel_size))

            mean_samp = fourier_Isamp[0].real / N
            amp_samp = 2 * np.abs(fourier_Isamp[k]) / N
            phase_samp = np.angle(fourier_Isamp[k])

            mean_ref = fourier_Iref[0].real / N
            amp_ref = 2 * np.abs(fourier_Iref[k]) / N
            phase_ref = np.angle(fourier_Iref[k])

            mean_list.append(mean_samp/mean_ref)
            centers.append(np.mean(x_seg))
            phi_list.append(phase_samp - phase_ref)

        return np.array(phi_list), np.array(mean_list)

def compute_phase_shift_lstsq(I_ref, I_tumor, I_no_tumor, num_realizations, photons):

    phi_tumor_results = []
    phi_no_tumor_results = []

    for i in range(num_realizations): 
        I_ref_noisy = np.random.poisson(I_ref * photons)
        I_tumor_noisy = np.random.poisson(I_tumor * photons)
        I_no_tumor_noisy = np.random.poisson(I_no_tumor * photons)
        phi_tumor = estimate_phi(I_ref_noisy, I_tumor_noisy)
        phi_no_tumor = estimate_phi(I_ref_noisy, I_no_tumor_noisy)
        phi_tumor_results.append(phi_tumor)
        phi_no_tumor_results.append(phi_no_tumor)

    phi_tumor_array = np.array(phi_tumor_results)
    phi_no_tumor_array = np.array(phi_no_tumor_results)
    return phi_tumor_array, phi_no_tumor_array

def compute_phase_shift_fourier(I_ref, I_tumor, I_no_tumor, num_realizations, photons):

    phi_tumor_results = []
    phi_no_tumor_results = []
    mean_no_tumor_results = []
    mean_tumor_results = []
    print ("photons per pixel",photons)
    for i in range(num_realizations): 
        I_ref_noisy = np.random.poisson(I_ref * photons)
        I_tumor_noisy = np.random.poisson(I_tumor * photons)
        I_no_tumor_noisy = np.random.poisson(I_no_tumor * photons)
        phi_tumor, mean_tumor = estimate_phi_fourier(I_ref_noisy, I_tumor_noisy)
        phi_no_tumor, mean_no_tumor = estimate_phi_fourier(I_ref_noisy, I_no_tumor_noisy)
        
        phi_tumor_results.append(phi_tumor)
        phi_no_tumor_results.append(phi_no_tumor)
        
        mean_tumor_results.append(mean_tumor)
        mean_no_tumor_results.append(mean_no_tumor)

    phi_tumor_array = np.array(phi_tumor_results)
    phi_no_tumor_array = np.array(phi_no_tumor_results)

    mean_tumor_array = np.array(mean_tumor_results)
    mean_no_tumor_array = np.array(mean_no_tumor_results)    
    
    return phi_tumor_array, phi_no_tumor_array, mean_tumor_array, mean_no_tumor_array

def compute_total_phase(phi_tumor,phi_no_tumor):
    phi_tumor = np.asarray(phi_tumor)
    phi_no_tumor = np.asarray(phi_no_tumor)
    # Ensure shape (num_realizations, n_features)
    if phi_tumor.ndim == 1:
        phi_tumor = phi_tumor[None, :]
    if phi_no_tumor.ndim == 1:
        phi_no_tumor = phi_no_tumor[None, :]
    total_phi_tumor = np.cumsum(phi_tumor, axis=1)
    total_phi_no_tumor = np.cumsum(phi_no_tumor, axis=1)
    return total_phi_no_tumor, total_phi_tumor

def compute_sdnr(G0, G1):
    """
    G0: array (N0, n) realizations under H0
    G1: array (N1, n) realizations under H1
    Returns: SDNR, SDNR_squared, weight_vector w, t_H0, t_H1
    """
    G0 = np.asarray(G0)
    G1 = np.asarray(G1)
    # Accept 1D inputs as single-feature observations
    if G0.ndim == 1:
        G0 = G0[:, None]
    if G1.ndim == 1:
        G1 = G1[:, None]

    # means
    g0_bar = np.mean(G0, axis=0)
    g1_bar = np.mean(G1, axis=0)
    d = g1_bar - g0_bar  # Δg

    # covariances (sample covariance; rows are samples)
    K0 = np.cov(G0, rowvar=False, bias=False)   # shape (n,n)
    K1 = np.cov(G1, rowvar=False, bias=False)

    Ksum = K0 + K1
    n = Ksum.shape[0]

    lam = 1e-6 * np.trace(Ksum) / n
    print("lam",lam)
    Kreg = Ksum + lam * np.eye(n)

    # compute weight vector w = (K0+K1)^{-1} Δg
    w = np.linalg.solve(Kreg, d)

    # SDNR^2 = d^T (K0+K1)^{-1} d
    sdnr2 = float(d.dot(w))
    sdnr = np.sqrt(max(sdnr2, 0.0))

    # test-statistic distributions (linear observer)
    t_H0 = G0.dot(w)   # shape (N0,)
    t_H1 = G1.dot(w)   # shape (N1,)

    return sdnr

def photons_vs_sdnr(I_ref, I_tumor, I_no_tumor, file_name, d_sphe):
    sdnr_phi_list = []
    sdnr_total_phi_list = []
    sdnr_mean_list = []
    photons_list = np.logspace(10, 3, num=40, base=10.0, dtype=int)[::-1]
    #photons_list = np.linspace(1e8, 1e15, num=40, dtype=int)
    for photons in photons_list:
        # compute phase-shifts for this photon count

        I_ref_2D_pixel, I_tumor_2D_pixel, I_no_tumor_2D_pixel = compute_intensity_2D_pixels(I_ref, I_tumor, I_no_tumor,d_sphe)
        phi_tumor_array, phi_no_tumor_array, mean_tumor_array, mean_no_tumor_array = compute_phase_shift_fourier(I_ref_2D_pixel, I_tumor_2D_pixel, 
                                                                          I_no_tumor_2D_pixel, num_realizations=500, photons=photons)
        #phi_tumor_array, phi_no_tumor_array = compute_phase_shift_fourier(I_ref, I_tumor, I_no_tumor,
                                                               # num_realizations=500, photons=photons)
        total_phi_no_tumor, total_phi_tumor = compute_total_phase(phi_tumor_array,phi_no_tumor_array)
        
        sdnr_phi_total = compute_sdnr(total_phi_no_tumor, total_phi_tumor)
        sdnr_phi = compute_sdnr(phi_no_tumor_array, phi_tumor_array)
        sdnr_mean = compute_sdnr(mean_no_tumor_array, mean_tumor_array)
        
        sdnr_total_phi_list.append(sdnr_phi_total)
        sdnr_phi_list.append(sdnr_phi)
        sdnr_mean_list.append(sdnr_mean)

    if not os.path.exists(file_name):
        # File does NOT exist — create it
        results = pd.DataFrame({
        "photons": photons_list,
        f"SDNR_{int(round(d_sphe))}um": sdnr_total_phi_list
        })
        results.to_csv(file_name, index=False)
    else:
        # File exists — load and add new column
        results = pd.read_csv(file_name)
        
        # Add new column (ensure lengths match)
        results[f"SDNR_{int(round(d_sphe))}um"] = sdnr_total_phi_list
        
        results.to_csv(file_name, index=False)


    return photons_list, sdnr_phi_list, sdnr_mean_list


#I_ref_2D_pixel, I_tumor_2D_pixel, I_no_tumor_2D_pixel = compute_intensity_2D_pixels(I_ref, I_tumor, I_no_tumor)
#phi_tumor_array, phi_no_tumor_array = compute_phase_shift_fourier(I_ref_2D_pixel, I_tumor_2D_pixel, I_no_tumor_2D_pixel, num_realizations=500, photons=num_ph)
#print(phi_tumor_array.shape)


d_sphss = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
for d_sph in d_sphss:
    col_name = f"{int(round(d_sph))}um"
    I_tumor = pd.read_csv(data)[col_name].to_numpy()
    I_ref = pd.read_csv(data)["I_ref"].to_numpy()
    I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()
    I_tumor = I_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
    I_no_tumor = I_no_tumor * np.exp(-mu_bkg_in_1_cm * additional_thickness_cm)
    photons_list, sdnr_list, sdnr_mean_list= photons_vs_sdnr(I_ref, I_tumor, I_no_tumor, "sdnr_results_60keV_all_sizes_scaled_12cm_2D_pixel_large_samples_217_total_phase_only_phase_v2.csv", d_sph)
