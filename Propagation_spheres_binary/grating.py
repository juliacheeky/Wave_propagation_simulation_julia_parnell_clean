import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import rotate
import scipy.fft
import skimage as ski
import xraylib as xrl
from tqdm import tqdm
from parameters import *

class Grating:                             
    
    def __init__(self, 
                 px_in_um: float) -> None:
        """
        Initializes an instance of the Grating class with the specified 
        parameters.

        This constructor sets up the common properties for both one- and two-
        dimensional gratings, including the grating period, duty cycle, 
        phase-shift, and grating material.

        Args:
            px_in_um (float): Grating period in the X direction, in um.
            py_in_um (float): Grating period in the Y direction, in um.
            dc (float): Grating duty cycle, defined as the ratio of the width 
                        of the open parts of the grating (fringes or holes) to
                        the total grating period.
            ph_shift_in_rad (float): Phase-shift of the grating, in rad.
            t_grat_in_um (float): Thickness of the grating, in um.
            mat_grat_type (str): Grating material type. The variable can only 
                                 have the values "element" or "compound".
            mat_grat (str): Grating material. If it is an element, write its
                            symbol. If it is a compound, write its chemical
                            formula. 
            rho_grat_in_g_cm3 (float): Mass density of the grating material, in
                                       g/cm3. If it is a compound, write its 
                                       mass density value (check the NIST 
                                       compound catalog).
                                       
        Using these arguments, the following variables are calculated:

            py_in_um (float): Grating period in the Y direction, in um.
            Z_grat (int): Atomic number of the grating material, if it is an
                          element.
            mu_grat_in_1_m (float): Linear attenuation coefficient of the 
                                    grating material, in 1/m.
            delta_grat (float): Phase coefficient of the grating material. 
        """
        self.px_in_um = px_in_um
        self.px_in_pix = int((self.px_in_um * 1e-6) / sim_pix_size_in_m)

    def create_grating(self) -> np.ndarray:
        """
        Creates a binary grating based on the specified period.

        Returns:
            np.ndarray: The binary grating array.
        """
        x_walk=np.linspace(0, img_size_in_pix, img_size_in_pix)
        E0_pi2_phase = np.where(np.mod(x_walk, self.px_in_pix) < self.px_in_pix / 2, -1/2, 0)
        grating_1d = np.exp(1j * np.pi * E0_pi2_phase)
        return grating_1d
