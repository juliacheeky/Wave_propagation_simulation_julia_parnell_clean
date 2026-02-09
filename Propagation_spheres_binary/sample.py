import numpy as np
import scipy.fft
from skimage.draw import disk
import xraylib as xrl
from typing import Tuple
from parameters import * 
import matplotlib.pyplot as plt

class Sample:                             
    
    def __init__(self, 
                 t_samp_in_mm: float, 
                 d_sph_in_um: float, 
                 f_sph: float,  
                 mat_sph_type: str, 
                 mat_bkg_type: str, 
                 mat_sph: str, 
                 mat_bkg: str,
                 rho_sph_in_g_cm3: float = None, 
                 rho_bkg_in_g_cm3: float = None) -> None:
        """
        Initializes an instance of the Sample class with the specified 
        parameters.

        Args:
            t_samp_in_mm (float): Sample thickness in mm.
            d_sph_in_um (float): Sphere diameter in um. 
            f_sph (float): Packing fraction of the spheres, defined as the 
                           ratio between the number of spheres and the total
                           sample volume. 
            mat_sph_type (str): Sphere material type. The variable can only 
                                have the values "element" or "compound".
            mat_bkg_type (str): Background material type. The variable can only 
                                have the values "element" or "compound".
            mat_sph (str): Sphere material. If it is an element, write its
                           symbol. If it is a compound, write its chemical
                           formula. 
            mat_bkg (str): Background material. If it is an element, write its
                           symbol. If it is a compound, write its chemical
                           formula. 
            rho_sph_in_g_cm3 (float, optional): Mass density of the sphere 
                                                material, in g/cm3. If it is a
                                                compound, write its mass density
                                                value. Defaults to None.
            rho_bkg_in_g_cm3 (float, optional): Mass density of the background 
                                                material, in g/cm3. If it is a
                                                compound, write its mass density
                                                value. Defaults to None.

        Raises:
            ValueError: If an invalid sphere or background material type is 
                        provided (neither "element" nor "compound").
            ValueError: If the mass density for a compound material (sphere or 
                        background) is not provided when needed.
        
        Using these arguments, the following variables are calculated:

            Z_sph (int): Atomic number of the sphere mateiral.
            mu_sph_in_1_m (float): Linear attenuation coefficient of the sphere
                                   material, in 1/m.
            delta_sph (float): Phase coefficient of the sphere material.
            Z_bkg (int): Atomic number of the background mateiral.
            mu_bkg_in_1_m (float): Linear attenuation coefficient of the 
                                   background material, in 1/m.                                  
            delta_bkg (float): Phase coefficient of the background material.

        """
                     
        self.thickness_in_mm = t_samp_in_mm
        self.d_sph_in_um = d_sph_in_um
        self.f_sph = f_sph

        #total sample thickness in pixels
        self.t_samp_in_pix = int((self.thickness_in_mm * 1e-3) / sim_pix_size_in_m)

        #number of slices the sample gets chopped up into
        self.num_slc = int(self.t_samp_in_pix /t_slc_in_pix)
        
        #diameter of spheres in pixels
        self.d_sph_in_pix = int((self.d_sph_in_um * 1e-6) / sim_pix_size_in_m)

        #radius of spheres in pixels
        self.r_sph_in_pix = int(self.d_sph_in_pix / 2)                     
        
        #total number of spheres in the sample (calculated from the particle fraction)
        self.num_sph_2dslice = int((samp_size_in_pix * self.t_samp_in_pix * f_sph)/(np.pi * self.r_sph_in_pix**2))
        
        # --- Getting sample properties from database ---------------------------------------------------

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

    def draw_sph_centers_2d(self, seed: int):
        """
        Creates the random centres of the disks within the sample 
        (also making sure they don't overlap too much) 
        
        Args:
            seed (int): Seed for the random number generator to ensure
                        reproducibility.
        """
        np.random.seed(seed) # set seed to allow reproducability

        # get twice as many a needed, so we can pick the furthest apart
        centres = np.random.randint(self.r_sph_in_pix, [self.t_samp_in_pix-self.r_sph_in_pix, samp_size_in_pix-self.r_sph_in_pix], (self.num_sph_2dslice*2, 2))
        # calculate distances^2 between all pairs of points
        distances = np.sum(np.square(centres.reshape((-1, 1, 2)) - centres), -1)
        # ignore values in lower half by setting them to max possible
        distances[np.arange(distances.shape[0])[:,None] >= np.arange(distances.shape[1])] = self.t_samp_in_pix*samp_size_in_pix
        # get the minumim distance to previous points
        min_distances = np.nanmin(distances, 0)
        # sort indices by descreasing distance, get best half
        indices = np.argsort(-min_distances)[0:self.num_sph_2dslice]
        return centres.take(indices, 0)


    def create_slice2d(self, 
                       seed: int) -> Tuple[np.ndarray, 
                                           np.ndarray]:
        """
        Generates th 2D sample containing the randomly distributed disks and its
        background.

        Args:
            seed (int): Seed for the random number generator to ensure
                        reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 2D binary
                                           arrays with spheres and the background
                                           of the slice, respectively.
        """

        # Create the 2D slice that will contain the disks
        slc2d_sph = np.zeros((samp_size_in_pix,self.t_samp_in_pix), dtype=np.uint16) 
        
        centres = self.draw_sph_centers_2d(seed)

        # Create a 2D sphere with the center at (cX, cZ)
        for z,x in centres:
            rr, cc = disk((x,z),self.r_sph_in_pix, shape=(samp_size_in_pix,self.t_samp_in_pix))
            slc2d_sph[rr, cc] = 1

        slc2d_sph_real = np.abs(slc2d_sph)
        
        # Create the background of the slice (inverse binary map)
        slc2d_bkg = np.ones(slc2d_sph_real.shape) - slc2d_sph_real
        return slc2d_sph_real, slc2d_bkg

    def create_projected_1d_slices(self,seed: int) -> Tuple[np.ndarray,np.ndarray]:
        """
        Takes the 2D sample and slices it into sections that are summed up so we can propegate through it
        in the thin slice aproximation. These individual vectors are thickness maps representing one section
        of the sample.
        Args:
            seed (int): Seed for the random number generator to ensure
                        reproducibility.        
        Returns:
            Tuple[np.ndarray, np.ndarray]:  A tuple containing the 1D projected 
                                            sphere and background slices ready for propagation
        """
        slice_profiles_sph = []
        slice_profiles_bkg = []
        slc2d_sph_real, slc2d_bkg = self.create_slice2d(seed=0)

        for i in range(self.num_slc):
            start = i * t_slc_in_pix
            end = start + t_slc_in_pix

            slice_chunk_sph = slc2d_sph_real[:,start:end] 
            slice_chunk_bkg = slc2d_bkg[:, start:end]

            profile_sph = np.sum(slice_chunk_sph, axis=1)           # sum over rows
            profile_bkg = np.sum(slice_chunk_bkg, axis=1)

            slice_profiles_sph.append(profile_sph)
            slice_profiles_bkg.append(profile_bkg)

        slice_profiles_sph = np.array(slice_profiles_sph)  # (num_slc, img_size_in_pix)
        slice_profiles_bkg = np.array(slice_profiles_bkg)
        
        # New slices with zero padding (I usually don't do any padding but it's here either way)
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
        Instead of having ones and zeros in the array we now have the complex numbers 
        correspning to the refractive indexes multiplied by the thickness of the structures.

        Args:
            samp_sph_in_m (np.ndarray): The positive projected slices (thickness maps)
            samp_bkg_in_m (np.ndarray): The negative/background projected slices (thickness maps)

        Returns:
            np.ndarray: Thickness maps with complex numbers in them ready for multiplication with wave vector
        """
        return np.exp(-1j * k_in_1_m * (self.delta_sph * \
                                samp_sph_in_m + self.delta_bkg * \
                                samp_bkg_in_m)) * \
               np.exp(-((self.mu_sph_in_1_m / 2) * samp_sph_in_m + \
                      (self.mu_bkg_in_1_m / 2) * samp_bkg_in_m)) 


# You can uncomment this if you want to generate and save the 2D binary array 
# with ones for spheres and zeros for background
"""
samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                d_sph_in_um = d_sph_in_um,
                f_sph = f_sph,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg, 
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 

sample_2d = samp2d.create_slice2d(seed=0)[0]
print(sample_2d.shape)
np.save("sample_2d_spheres.npy", sample_2d)
"""

