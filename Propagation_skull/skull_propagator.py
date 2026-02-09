import numpy as np
from tqdm import tqdm
from skull_grating import *
from skull_sample import *
from skull_parameter import *

"""I decided to assign the material properties to the propagator since they are needed there for the sample interaction and are energy dependent"""

class Propagator:

    def __init__(self, 
                 grat: Grating, 
                 samp: Sample_Skull,  # muss zu Skull_sample gewechselt werden fue skull
                 prop_in_m: float,
                 mat_bone: float,
                 mat_air:float,
                 energy:float,
                 rho_bone_in_g_cm3:float,
                 rho_air_in_g_cm3:float
                 ) -> None:
        """
        Initializes an instance of the wave field class with the specified 
        parameters.

        Args:
            grat (Grating): The grating object used in the setup.
            samp (Sample): The sample object being analyzed.
            det (Detector): The detector object capturing the wave field.

        Using these arguments, the following variables are calculated:

            talbot_in_m (float): Talbot distance, in m.
            grat2det_in_m (float): Grating-to-detector distance, in m.
            grat2samp_in_m (float): Grating-to-sample distance, in m.
            samp2det_in_m (float): Sample-to-detector distance, in m.
        """
        
        self.grat = grat
        self.samp = samp 
        self.prop_in_m = prop_in_m
        self.mat_bone = mat_bone
        self.mat_air = mat_air
        self.energy = energy
        self.rho_bone_in_g_cm3 = rho_bone_in_g_cm3
        self.rho_air_in_g_cm3 = rho_air_in_g_cm3

        self.l_in_m = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (energy * 1e3) 
        self.k_in_1_m = 2 * np.pi * (energy * 1e3) / \
        (cte.physical_constants["Planck constant in eV s"][0] * cte.c)
        self.talbot_in_m = 2 * (grat.px_in_um * 1e-6)**2 / self.l_in_m 
        self.grat2det_in_m = 1/4 * self.talbot_in_m  
        #self.grat2det_in_m = 0.8
        self.grat2samp_in_m = self.grat2det_in_m - self.prop_in_m - \
                              (samp.thickness_in_mm * 1e-3) / 2
        self.samp2det_in_m = self.prop_in_m - (samp.thickness_in_mm * 1e-3) / 2
        self.bin_grat = grat.create_grating()

        self.mu_bone_in_1_m = xrl.CS_Total_CP(self.mat_bone, self.energy) * self.rho_bone_in_g_cm3 * 100
        self.delta_bone = 1 - xrl.Refractive_Index_Re(self.mat_bone,self.energy,self.rho_bone_in_g_cm3)
        print(f"Bone: Delta: {self.delta_bone:.3e}, Mu 1/m: {self.mu_bone_in_1_m:.3e} 1/m at {self.energy} keV")
        self.mu_air_in_1_m = xrl.CS_Total_CP(self.mat_air, self.energy) * self.rho_air_in_g_cm3 * 100
        self.delta_air = 1 - xrl.Refractive_Index_Re(self.mat_air,self.energy,self.rho_air_in_g_cm3)

    # --- Basic operations ----------------------------------------------------

    def inter_wavefld_grat(self, 
                           wavefld: np.ndarray) -> np.ndarray:
        """
        Applies the grating interaction to the wave field.

        This method modifies the input wave field based on the phase shift
        introduced by the grating and the attenuation due to the grating 
        material properties.

        Args:
            wavefld (np.ndarray): The input wave field to be modified.
            bin_grat (np.ndarray): The binary grating.

        Returns:
            np.ndarray: The modified wave field after interaction with the 
                        grating.
        """

        return wavefld * self.bin_grat  
    
    def inter_wavefld_samp(self, 
                           wavefld: np.ndarray, 
                           samp_sph_in_m: np.ndarray,
                           samp_bkg_in_m: np.ndarray) -> np.ndarray:
        """
        Applies the sample interaction to the wave field.

        This method modifies the input wave field based on the phase shifts 
        and attenuation resulting from the sample properties.

        Args:
            wavefld (np.ndarray): The input wave field to be modified.
            samp_sph_in_m (np.ndarray): _description_
            samp_bkg_in_m (np.ndarray): _description_

        Returns:
            np.ndarray: The modified wave field after interaction with the 
                        sample.
        """
        return wavefld * np.exp(-1j * self.k_in_1_m * (self.delta_bone * \
                                samp_sph_in_m + self.delta_air * \
                                samp_bkg_in_m)) * \
               np.exp(-((self.mu_bone_in_1_m / 2) * samp_sph_in_m + \
                      (self.mu_air_in_1_m / 2) * samp_bkg_in_m)) 
    
    def create_Fresnel_kernel(self, z_in_m: float) -> np.ndarray:
        """
        Creates a Fresnel propagation kernel in the Fourier domain to simulate 
        wave field propagation over a certain distance in the Fourier space.

        Args:
            z_in_m (float): The propagation distance, in m.

        Returns:
            np.ndarray: The Fresnel kernel for the specified propagation 
                        distance.
        """

        u = scipy.fft.fftfreq(img_size_in_pix, d=sim_pix_size_in_m) #removed the 2 pi here... is that ok?
        return np.exp(1j*(2 * np.pi / self.l_in_m) * z_in_m) * np.exp(-1j * np.pi * self.l_in_m * z_in_m * (u**2))

    def prop_wavefld(self, 
                    wavefld: np.ndarray, 
                    kernel: np.ndarray) -> np.ndarray:  
        """
        Propagates the wave field over a specified distance using the Fresnel 
        kernel.

        Args:
            wavefld (np.ndarray): The input wave field to be propagated.
            kernel (float): The Fresnel kernel in Fourier.

        Returns:
            np.ndarray: The propagated wave field after applying the Fresnel 
                        propagation.
        """

        return scipy.fft.ifftn(scipy.fft.fftn(wavefld) * kernel)  
    
    # --- Create Iref and Isamp -----------------------------------------------    

    def obtain_Iref_Isamp(self, 
                          wavefld: np.ndarray, 
                          bin_grat: np.ndarray) -> Tuple[np.ndarray, 
                                                         np.ndarray]:
        """
        Obtains a single reference and sample intensity image by shifting the 
        grating in both X and Y directions. This function is designed to be 
        used with the multiprocessing library to run all wave propagation 
        simulations in parallel.

        Args:
            wavefld (np.ndarray): The input wave field to be processed.
            bin_grat (np.ndarray):The binary grating.
            num_steps_in_pix (int): The number of steps per direction (X or Y)
                                    for grating shifting, in pix. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing a single 
                                           reference and sample intensity 
                                           image, respectively.
        """
        
        # Obtain the Fresnel kernels for each propagation distance
        kernel_grat2det = self.create_Fresnel_kernel(self.grat2det_in_m) 
        kernel_grat2samp = self.create_Fresnel_kernel(self.grat2samp_in_m) 
        kernel_samp2det = self.create_Fresnel_kernel(self.samp2det_in_m) 
        kernel_slc2slc = self.create_Fresnel_kernel(t_slc_in_pix * \
                                                    sim_pix_size_in_m)

        # Interaction of the initial wave field with the binary grating
        wavefld_ag = self.inter_wavefld_grat(wavefld)

        # --- Obtain Iref -----------------------------------------------------

        # Propagation of the wave field after the grating until the detector 
        # position
        wavefld_prop = self.prop_wavefld(wavefld_ag, kernel_grat2det) 
        del kernel_grat2det

        # Intensity of the wave fields in the detector plane
        Iref_large = np.abs(wavefld_prop)**2
    
        
        # --- Obtain Isamp ----------------------------------------------------

        # Propagation of the wave field after the grating until the sample 
        # front position
        wavefld_bs = self.prop_wavefld(wavefld_ag, kernel_grat2samp) 
        del kernel_grat2samp

        slice_profiles_path = "slices_data.npz"
        data = np.load(slice_profiles_path)
        self.slc1d_bone = data['slc2d_sph_padded']
        self.slc1d_pores = data['slc2d_bkg_padded']   
        print(self.slc1d_bone.shape)
        print(f"wavelength: {self.l_in_m} at energy: {self.energy} keV")
        
        for i in tqdm(range(self.samp.num_slc)):           
            # Creation of the sample slice

            # Interaction between the wave field before the slice and the sample
            # slice
            wavefld_as = self.inter_wavefld_samp(wavefld_bs, 
                                                 self.slc1d_bone[i,:] * sim_pix_size_in_m,
                                                 self.slc1d_pores[i,:] * sim_pix_size_in_m)
            # Propagation of the wave field after the sample slice until the 
            # next sample slice
            wavefld_prop = self.prop_wavefld(wavefld_as, kernel_slc2slc)

            # The propagated wave field becomes the wave field before the next
            # sample slice
            wavefld_bs = wavefld_prop
        
        del kernel_slc2slc

        # The last value of 'wavefld_bs' is the wave field after the whole sample
        wavefld_as = wavefld_bs
        del wavefld_bs

        # Propagation of the wave field after the sample until the detector 
        # position
        wavefld_prop = self.prop_wavefld(wavefld_as, kernel_samp2det)
        del wavefld_as, kernel_samp2det

        # Intensity of the wave fields in the detector plane
        Isamp_large = np.abs(wavefld_prop)**2

        return Iref_large, Isamp_large 