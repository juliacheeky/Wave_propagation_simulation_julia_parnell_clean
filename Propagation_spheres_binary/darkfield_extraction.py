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
