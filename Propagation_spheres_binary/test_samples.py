import numpy as np
import scipy.fft
from skimage.draw import disk
import xraylib as xrl
from typing import Tuple
from parameters import * 
import matplotlib.pyplot as plt

class test_Sample:                             
    
    def __init__(self, 
                 t_samp_in_mm: float, 
                 mat_sph_type: str, 
                 mat_bkg_type: str, 
                 mat_sph: str, 
                 mat_bkg: str,
                 rho_sph_in_g_cm3: float = None, 
                 rho_bkg_in_g_cm3: float = None) -> None:

                     
        self.thickness_in_mm = t_samp_in_mm
        self.t_samp_in_pix = int((self.thickness_in_mm * 1e-3) / sim_pix_size_in_m)
        self.num_slc = int(self.t_samp_in_pix /t_slc_in_pix)
        self.mat_sph = mat_sph 
        if(mat_sph_type == "element"): 
            self.Z_sph = xrl.SymbolToAtomicNumber(self.mat_sph)
            self.rho_sph_in_g_cm3 = xrl.ElementDensity(self.Z_sph)
            self.mu_sph_in_1_m = xrl.CS_Total(self.Z_sph, E_in_keV) * \
                                 self.rho_sph_in_g_cm3 * 100
        elif(mat_sph_type == "compound"): 
            self.rho_sph_in_g_cm3 = rho_sph_in_g_cm3
            if(self.rho_sph_in_g_cm3 == None):
                raise ValueError("Please provide a valid mass density value " \
                                 "for the sphere compound")
            self.mu_sph_in_1_m = xrl.CS_Total_CP(self.mat_sph, E_in_keV) \
                                 * self.rho_sph_in_g_cm3 * 100
        else:
            raise ValueError("Sphere material type not defined")       
        self.delta_sph = 1 - xrl.Refractive_Index_Re(self.mat_sph, 
                                                     E_in_keV,
                                                     self.rho_sph_in_g_cm3)
        self.mat_bkg = mat_bkg
        if(mat_bkg_type == "element"):
            self.Z_bkg = xrl.SymbolToAtomicNumber(self.mat_bkg)
            self.rho_bkg_in_g_cm3 = xrl.ElementDensity(self.Z_bkg)
            self.mu_bkg_in_1_m = xrl.CS_Total(self.Z_bkg, E_in_keV) * \
                                 self.rho_bkg_in_g_cm3 * 100
        elif(mat_bkg_type == "compound"): 
            self.rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3
            if(self.rho_bkg_in_g_cm3 == None):
                raise ValueError("Please provide a valid mass density value " \
                                 "for the background compound")
            self.mu_bkg_in_1_m = xrl.CS_Total_CP(self.mat_bkg, E_in_keV) \
                                 * self.rho_sph_in_g_cm3 * 100
        else:
            raise ValueError("Background material type not defined")
        self.delta_bkg = 1 - xrl.Refractive_Index_Re(self.mat_bkg, 
                                                     E_in_keV,
                                                     self.rho_bkg_in_g_cm3)
    
    # --- Spheres -------------------------------------------------------------

    import numpy as np

    def create_slice2d(self, thickness=1, start=0, dtype=int):
        """
        Alternating horizontal stripes of 0/1 with controllable stripe thickness.
        thickness = number of consecutive rows per stripe.
        start = 0 or 1 (value of the first stripe)
        """
        stripe_ids = np.arange(self.t_samp_in_pix) // thickness          # 0,0,0,1,1,1,2,2,2,...
        rows = (stripe_ids % 2)                              # 0/1 alternating by stripe
        rows = rows ^ (start & 1)   
        sph_2d_samp = np.repeat(rows[:, None], img_size_in_pix, axis=1).astype(dtype)       # optionally start with 1
        bkg_2d_samp = 1 - sph_2d_samp
        return sph_2d_samp.T, bkg_2d_samp.T


    def create_projected_1d_slices(self,seed: int) -> Tuple[np.ndarray,np.ndarray]:
        slice_profiles_sph = []
        slice_profiles_bkg = []
        #slc2d_sph_real, slc2d_bkg = self.create_slice2d(seed=0)
        slc2d_sph_real, slc2d_bkg = self.create_slice2d(thickness=int(self.t_samp_in_pix/10), start=0, dtype=int)
        for i in range(self.num_slc):
            start = i * t_slc_in_pix
            end = start + t_slc_in_pix
            slice_chunk_sph = slc2d_sph_real[:,start:end] 
            slice_chunk_bkg = slc2d_bkg[:, start:end]          # select slice
            profile_sph = np.sum(slice_chunk_sph, axis=1)           # sum over rows
            profile_bkg = np.sum(slice_chunk_bkg, axis=1)
            slice_profiles_sph.append(profile_sph)
            slice_profiles_bkg.append(profile_bkg)
        slice_profiles_sph = np.array(slice_profiles_sph)  # (num_slc, img_size_in_pix)
        slice_profiles_bkg = np.array(slice_profiles_bkg)
        
        # New slices with zero padding
        slc2d_sph_padded = np.zeros((slice_profiles_sph.shape[0], img_size_in_pix))
        slc2d_bkg_padded = np.zeros((slice_profiles_bkg.shape[0], img_size_in_pix))
        x_offset = int((img_size_in_pix - samp_size_in_pix) / 2)
        slc2d_sph_padded[:, x_offset:x_offset + samp_size_in_pix] = slice_profiles_sph
        slc2d_bkg_padded[:, x_offset:x_offset + samp_size_in_pix] = slice_profiles_bkg
        return slc2d_sph_padded, slc2d_bkg_padded

    def samp_with_refract_property(self, 
                           samp_sph_in_m: np.ndarray,
                           samp_bkg_in_m: np.ndarray) -> np.ndarray:
        """
        Applies the refractive properties of the sample to the binary array.

        This method modifies the input wave field based on the phase shifts 
        and attenuation resulting from the sample properties.

        Args:
            samp_sph_in_m (np.ndarray): _description_
            samp_bkg_in_m (np.ndarray): _description_

        Returns:
            np.ndarray: The modified wave field after interaction with the 
                        sample.
        """
        return np.exp(-1j * k_in_1_m * (self.delta_sph * \
                                samp_sph_in_m + self.delta_bkg * \
                                samp_bkg_in_m)) * \
               np.exp(-((self.mu_sph_in_1_m / 2) * samp_sph_in_m + \
                      (self.mu_bkg_in_1_m / 2) * samp_bkg_in_m)) 

"""
samp2d = test_Sample(t_samp_in_mm = t_samp_in_mm,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg, 
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 

sample_2d = samp2d.create_slice2d( 
                                 thickness=int(samp2d.t_samp_in_pix/10), 
                                 start=1, 
                                 dtype=int)[0]
print(sample_2d.shape)
plt.imshow(sample_2d[::100, ::100], cmap="gray", aspect="auto")
plt.colorbar()
plt.savefig("test_strippy_sample.png", dpi=300, bbox_inches='tight')


#np.save("sample_2d_test.npy", sample_2d)
"""

