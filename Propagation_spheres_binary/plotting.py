import matplotlib.pyplot as plt
import numpy as np
from grating import *
from sample import *
from propagator import *
from parameters import *
from detector import *
import os
import pandas as pd
import time


def plot_intensity_withG2(det, prop,  wavefld_bg, save_plot=True):  
    
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
    print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV")
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {E_in_keV:.1f} keV")
    plt.plot(Iref_stepped, label='Iref with G2', linewidth=0.5, color='red')
    plt.plot(Isamp_stepped, label='Isamp with G2', linewidth=0.5, color='blue')
    #plt.title(f"Intensity Profile at {E_in_keV:.1f} keV | Visibility with sample: {visibility.real:.3f} \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    #plt.title(f"Intensity Profile at {E_in_keV:.1f} keV no G2 \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    plt.xlabel('Pixels')
    plt.xlim(1000, 1200)
    plt.ylabel('Intensity')
    plt.legend()

    if save_plot:
        #path_image = os.path.join("images", "bone_like_1" ,f"intensity_withG2_with_absorb_2mat_{name_mat_sph}_{name_mat_bkg}_{E_in_keV:.1f}keV_{t_samp_in_mm:.1f}mm.pdf")
        path_image = os.path.join("clossser_look_test.pdf")
        plt.savefig(path_image, dpi=600, bbox_inches='tight')
        time.sleep(5)
    del Iref_stepped, Isamp_stepped

def plot_intensity_withoutG2(det, prop,  wavefld_bg, save_plot=True): 
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, prop.bin_grat)
    x_walk=np.linspace(0, img_size_in_pix, img_size_in_pix)
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)
    df = pd.DataFrame({
        "x_detector":x_walk,
        "Iref_noG2": Iref_large,
        "Isamp_noG2": Isamp_large
    })

    df.to_csv("moa_SiO2_H2O/Iref_Isamp_noG2_40keV_6um_diam.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axs[0].plot(Iref_stepped, label='Iref with G2', linewidth=0.5, color='red')
    axs[0].plot(Isamp_stepped, label='Isamp with G2', linewidth=0.5, color='blue')
    axs[0].set_title("With G2")
    axs[0].set_xlabel('Pixels')
    axs[0].set_ylabel('Intensity')
    axs[0].legend()

    axs[1].plot(Iref_large, label='Iref', linewidth=0.5, color='red')
    axs[1].plot(Isamp_large, label='Isamp', linewidth=0.5, color='blue')
    axs[1].set_title("Without G2")
    axs[1].set_xlabel('Pixels')
    axs[1].legend()
    plt.tight_layout()
    plt.title(f'Energy: {E_in_keV:.1f} keV')

    plt.savefig('20keV_noG2_0175m_3diam', dpi=600, bbox_inches='tight')


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
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {E_in_keV:.1f} keV")
    return visibility, epsilon

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

def calc_vis_with_G2(det):
    df = pd.read_csv("/users/Parnell/Julias_code/moa_SiO2_H2O/Iref_Isamp_noG2_40keV_6um_diam.csv")
    Iref_large = df['Iref_noG2']
    Isamp_large = df['Isamp_noG2']
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)

    start_idx = int(0.1 * len(Iref_stepped))
    end_idx = int(0.9 * len(Iref_stepped))
    Iref_stepped = Iref_stepped[start_idx:end_idx]
    Isamp_stepped = Isamp_stepped[start_idx:end_idx]
    
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
    return visibility_s/visibility_r


"""
ft_Isamp_stepped = np.fft.fft(Isamp_stepped)
f0, f1 = ft_Isamp_stepped[0], ft_Isamp_stepped[128]
N = len(Isamp_stepped)
print("f[N//2] magnitude:", np.abs(ft_Isamp_stepped[N//2]))
print("f[0] magnitude:", np.abs(ft_Isamp_stepped[0]))
print("f[1] magnitude:", np.abs(ft_Isamp_stepped[1]))
print("f[128] magnitude:", np.abs(ft_Isamp_stepped[128]))


unique = sorted(np.abs(ft_Isamp_stepped))
if len(unique) < 2:
    print("No second largest value.")
else:
    second_largest = unique[-2]
    index = np.where(ft_Isamp_stepped == second_largest)[0]
    print(f"Second largest value: {second_largest}, at index: {index}")

magnitudes = np.abs(ft_Isamp_stepped)
unique_magnitudes = np.unique(magnitudes)

if len(unique_magnitudes) < 2:
    print("No second largest value.")
else:
    second_largest_mag = unique_magnitudes[-2]

    # Get all indices where magnitude matches the second largest
    indices = np.where(magnitudes == second_largest_mag)[0]

    print(f"Second largest magnitude: {second_largest_mag}, at index/indices: {indices}")

amplitude_f128 = 2 * np.abs(ft_Isamp_stepped[128]) / N
mean_value = np.real(f0/len(Isamp_stepped))
print(amplitude_f128)
vis = amplitude_f128 / mean_value
print(vis)
"""