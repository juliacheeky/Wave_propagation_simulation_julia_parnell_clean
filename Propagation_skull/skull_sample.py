import numpy as np
import scipy.fft
import xraylib as xrl
from typing import Tuple
import cv2  
from skull_parameter import * 
import matplotlib.pyplot as plt

"""What this currently does is only create the 1D slices from an array of ony zeros and ones"""

class Sample_Skull:                             
    
    def __init__(self, 
                 mat_bone: str, 
                 mat_air: str
                 ) -> None:

        self.mat_bone = mat_bone 
        self.mat_air = mat_air
        image = cv2.imread("/users/Parnell/Julias_code/raw_skull_images/04-02__rec_Tra0548.bmp", cv2.IMREAD_GRAYSCALE)
        pixel_size_skull = 6e-6
        ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8) 
        binary_cropped = binary[1000:2400,900:2300]
        scale_factor = pixel_size_skull / sim_pix_size_in_m

        self.positive = cv2.resize(
            binary_cropped,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )
        t_samp_in_pix = self.positive.shape[1]
        self.thickness_in_mm = t_samp_in_pix * sim_pix_size_in_m * 1e3
        self.num_slc = int(t_samp_in_pix /t_slc_in_pix) 
        print(f"Sample thickness: {self.thickness_in_mm:.2f} mm with {self.num_slc} slices")
        #self.rho_bone_in_g_cm3 = rho_bone_in_g_cm3
        #self.rho_air_in_g_cm3 = rho_air_in_g_cm3


    def create_slice2d(self)-> Tuple[np.ndarray, 
                                           np.ndarray]:

        negative = np.ones(self.positive.shape) - self.positive
        return self.positive, negative
    
    
    def create_projected_1d_slices(self) -> Tuple[np.ndarray,np.ndarray]:

 
        slice_profiles_sph = []
        slice_profiles_bkg = []
        bone, pores = self.create_slice2d()
        
        for i in range(self.num_slc):
            start = i * t_slc_in_pix
            end = start + t_slc_in_pix
            slice_chunk_sph = bone[:,start:end] 
            slice_chunk_bkg = pores[:, start:end]          # select slice
            profile_sph = np.sum(slice_chunk_sph, axis=1)           # sum over rows
            profile_bkg = np.sum(slice_chunk_bkg, axis=1)
            slice_profiles_sph.append(profile_sph)
            slice_profiles_bkg.append(profile_bkg)

        return np.array(slice_profiles_sph), np.array(slice_profiles_bkg)



"""samp2d = Sample_Skull(mat_bone = mat_bone, mat_air = mat_air) 
profiles_bones, profiles_pores = samp2d.create_projected_1d_slices()

positive, negative = samp2d.create_slice2d()
print(positive.shape)
profiles_bones, profiles_pores = samp2d.create_projected_1d_slices()
print(profiles_bones.shape)


plt.imshow(positive, cmap='gray', aspect='equal')

# Set tick positions (in pixels)
tick_spacing = 400  # adjust as needed
x_ticks = np.arange(0, positive.shape[1], tick_spacing)
y_ticks = np.arange(0, positive.shape[0], tick_spacing)

# Set tick labels in mm
plt.xticks(x_ticks, [f"{x*0.001:.1f}" for x in x_ticks])
plt.yticks(y_ticks, [f"{y*0.001:.1f}" for y in y_ticks])
plt.xlabel("mm")
plt.ylabel("mm")

plt.grid(True, color='red', linestyle='-', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.tight_layout()
plt.savefig("loeschen.pdf", dpi=300)"""
