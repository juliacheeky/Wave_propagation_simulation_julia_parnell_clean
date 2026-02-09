import matplotlib.pyplot as plt
import numpy as np
from skull_grating import *
from skull_sample import *
from skull_propagator import *
from skull_parameter import *
from skull_detector import *
import os
import pandas as pd
import time


def plot_intensity_withG2(energy,det, prop,  wavefld_bg, save_plot=True):  
    
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, prop.bin_grat)
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)

    I_max_samp = np.max(Isamp_stepped)
    I_min_samp = np.min(Isamp_stepped)
    I_max_ref = np.max(Iref_stepped)
    I_min_ref = np.min(Iref_stepped)
    a_1s = (I_max_samp-I_min_samp)/2
    a_1r = (I_max_ref-I_min_ref)/2
    a_0s = np.mean(Isamp_stepped.real)
    a_0r = np.mean(Iref_stepped.real)
    visibility_s = a_1s.real/a_0s
    visibility_r = a_1r.real/a_0r
    visibility = visibility_s/visibility_r
    #epsilon = -np.log(visibility) / (t_samp_in_mm * 1e-3)
    print(f"Visibility with sample: {visibility.real:.3f} at Energy: {energy:.1f} keV")
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {energy:.1f} keV")

    #plt.title(f"Intensity Profile at {E_in_keV:.1f} keV | Visibility with sample: {visibility.real:.3f} \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    #plt.title(f"Intensity Profile at {E_in_keV:.1f} keV no G2 \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].plot(Iref_stepped, color='blue')
    axs[0].plot(Isamp_stepped, color='red')
    axs[0].set_title(f"Intensities with G2")
    axs[0].set_xlabel("Pixels")
    axs[0].set_ylabel("Value")
    xmin, xmax = 50000, 52000
    axs[0].set_xlim(xmin, xmax)
    #axs[0].set_ylim(0, 1)
    #y1, y2 = 0.668, 0.034
    #axs[0].axhline(y=y1, color='black', linestyle='--', linewidth=1)
    #axs[0].axhline(y=y2, color='black', linestyle='-.', linewidth=1)

    axs[1].plot(Iref_large, color='blue')
    axs[1].plot(Isamp_large, color='red')
    axs[1].set_title(f"Intensities without G2")
    axs[1].set_xlabel("Pixels")
    axs[1].set_xlim(xmin, xmax)
    plt.legend()
    print(f"I_max_samp with G2: {I_max_samp:.3f}, I_min_samp with G2: {I_min_samp:.3f}")

    if save_plot:
        #path_image = os.path.join("images", "bone_like_1" ,f"intensity_withG2_with_absorb_2mat_{name_mat_sph}_{name_mat_bkg}_{E_in_keV:.1f}keV_{t_samp_in_mm:.1f}mm.pdf")
        path_image = os.path.join("clossser_look_test.pdf")
        plt.savefig(path_image, dpi=600, bbox_inches='tight')
    del Iref_stepped, Isamp_stepped


def save_visibility_epsilon(det, prop,  wavefld_bg, bin_grat,thick_samp_mm):
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, bin_grat)
    
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)
    Iref_stepped = Iref_stepped[int(img_size_in_pix*0.1):int(img_size_in_pix*0.9)]
    Isamp_stepped = Isamp_stepped[int(img_size_in_pix*0.1):int(img_size_in_pix*0.9)]
    I_max_samp = np.max(Isamp_stepped)
    I_min_samp = np.min(Isamp_stepped)
    I_max_ref = np.max(Iref_stepped)
    I_min_ref = np.min(Iref_stepped)
    a_1s = (I_max_samp-I_min_samp)/2
    a_1r = (I_max_ref-I_min_ref)/2
    a_0s = np.mean(Isamp_stepped.real)
    a_0r = np.mean(Iref_stepped.real)
    visibility_s = a_1s.real/a_0s
    visibility_r = a_1r.real/a_0r
    visibility = visibility_s/visibility_r
    epsilon = -np.log(visibility) / (thick_samp_mm * 1e-3)
    mean_transmission = a_0s/a_0r
    return visibility, epsilon, mean_transmission

def save_visibility_epsilon_sections(det, prop,  wavefld_bg, bin_grat,thick_samp_mm):
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, bin_grat)
    
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)
    #Iref_stepped = Iref_stepped[int(img_size_in_pix*0.1):int(img_size_in_pix*0.9)]
    #Isamp_stepped = Isamp_stepped[int(img_size_in_pix*0.1):int(img_size_in_pix*0.9)]

    vis_list = []
    epsilon_list = []
    x_walk = np.arange(len(Iref_stepped))

    for start in range(segment_size_in_pix,
                   len(x_walk) - segment_size_in_pix,
                   segment_size_in_pix):
        x_seg = x_walk[start:start+segment_size_in_pix]
        Iref_seg = Iref_stepped[start:start+segment_size_in_pix]
        Isamp_seg = Isamp_stepped[start:start+segment_size_in_pix]

        I_max_samp_seg = np.max(Isamp_seg)
        I_min_samp_seg = np.min(Isamp_seg)
        I_max_ref_seg = np.max(Iref_seg)
        I_min_ref_seg = np.min(Iref_seg)
        a_1s = (I_max_samp_seg-I_min_samp_seg)/2
        a_1r = (I_max_ref_seg-I_min_ref_seg)/2
        a_0s = np.mean(Isamp_seg.real)
        a_0r = np.mean(Iref_seg.real)
        visibility_s = a_1s.real/a_0s
        visibility_r = a_1r.real/a_0r
        visibility_seg = visibility_s/visibility_r
        epsilon_seg = -np.log(visibility_seg) / (thick_samp_mm * 1e-3)
        vis_list.append(visibility_seg)
        epsilon_list.append(epsilon_seg)
        print(f"Mean with sample: {a_0s:.3f}, Visibility: {visibility_seg:.3f} at Energy: {E_in_keV:.1f} keV")
    visibility = np.mean(vis_list)
    epsilon = np.mean(epsilon_list)
    return visibility, epsilon

"""def fourier_fit_vis(prop,  wavefld_bg, bin_grat,thick_samp_mm):
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, bin_grat)
    N_ref = len(Iref_large)
    N_samp = len(Isamp_large)
    x=np.linspace(0, img_size_in_pix, img_size_in_pix)*sim_pix_size_in_m
    N=(x[N_ref-1]-x[0])/(px_in_um*1e-6)
    k = 2* np.pi / (px_in_um*1e-6)

    I0_ref = (1 / N_ref) * np.sum(Iref_large) #medelvärdet (DC-nivån) för respektive signal.
    I0_samp = (1 / N_samp) * np.sum(Isamp_large)

    C_ref = (2 / N_ref) * np.sum(Iref_large * np.cos(k * x))
    C_samp = (2 / N_samp) * np.sum(Isamp_large * np.cos(k * x))

    S_ref = (2 / N_ref) * np.sum(Iref_large * np.sin(k * x))
    S_samp = (2 / N_samp) * np.sum(Isamp_large * np.sin(k * x))

    A1ref = np.sqrt(C_ref**2 + S_ref**2)
    A1samp = np.sqrt(C_samp**2 + S_samp**2)

    Vref = A1ref / I0_ref
    Vsamp = A1samp / I0_samp
    Isamp_fitted = I0_samp + C_samp * np.cos(k * x) + S_samp * np.sin(k * x)
    plt.plot(x, Iref_large, label='Iref_large')
    plt.plot(x, Isamp_large, label='Isamp_large')
    plt.plot(x, Isamp_fitted, label='Isamp_fitted', linestyle='--')
    plt.legend()
    plt.xlim(0.005, 0.0052)
    plt.savefig("fit_look_fourier", dpi=600, bbox_inches='tight')
    visibility = Vsamp / Vref
    epsilon = epsilon = -np.log(visibility) / (thick_samp_mm * 1e-3)
    return visibility, epsilon"""


def plot_epsilon_vs_d():
    df = pd.read_csv('visibility_results_2.csv')  # Replace with the actual path if needed

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['Sphere size (um)'], df['Epsilon'], marker='o', linestyle='-')

    # Labels and title
    plt.xlabel('Sphere Size (μm)')
    plt.ylabel('Epsilon')
    plt.title('Epsilon vs Sphere Diameter')

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("epsilon_diam.pdf", dpi=300, bbox_inches='tight')

def plot_single_slice_pair(slc2d_sph_padded, slc2d_bkg_padded, slice_idx=0, save_plot=True):

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].plot(slc2d_sph_padded[slice_idx], color='blue')
    axs[0].set_title(f"Slice {slice_idx} (Spheres)")
    axs[0].set_xlabel("Pixels")
    axs[0].set_ylabel("Value")

    axs[1].plot(slc2d_bkg_padded[slice_idx], color='orange')
    axs[1].set_title(f"Slice {slice_idx} (Background)")
    axs[1].set_xlabel("Pixels")

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"slice_pair_{slice_idx}.pdf", dpi=300, bbox_inches='tight')
