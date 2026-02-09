from grating import *
from sample import *
from propagator import *
import numpy as np
from parameters import *
from detector import *
from plotting import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os

samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                d_sph_in_um = d_sph_in_um,
                f_sph = f_sph,
                mat_sph_type = mat_sph_type,
                mat_bkg_type = mat_bkg_type,
                mat_sph = mat_sph, 
                mat_bkg = mat_bkg,
                rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3) 


delta_delta = samp2d.rho_bkg_in_g_cm3 - samp2d.rho_sph_in_g_cm3
delta_mu = samp2d.mu_bkg_in_1_m - samp2d.mu_sph_in_1_m
corr_length = (l_in_m*prop_in_m)/(px_in_um * 1e-6) 

mu_bkg_in_1_m = xrl.CS_Total_CP(mat_bkg, E_in_keV) * rho_bkg_in_g_cm3 * 100 
print(mu_bkg_in_1_m)
def mu_d(lambda_, f, delta_chi, d, D):

    D_prime = D / d
    prefactor = (3 * np.pi**2 / lambda_**2) * f * abs(delta_chi)**2 * d
    print(f"D_prime: {D_prime:.3f}, prefactor: {prefactor:.3e}, delta_chi: {abs(delta_chi)**2:.3e}")
    if D > d:
        sqrt_term = np.sqrt(D_prime**2 - 1)
        shape_factor = (D_prime - sqrt_term * (1 + 1 / (2 * D_prime**2))+ (1 / D_prime - 1 / (4 * D_prime**3))) * np.log((D_prime + sqrt_term) / (D_prime - sqrt_term))
    else:
        shape_factor = D_prime
    print(f"Shape factor: {shape_factor:.3f}")
    return prefactor * shape_factor

#mu_d = mu_d(l_in_m, f_sph, delta_chi, corr_length, 10e-6)


def mu_d_normalisiert(D_prime):
    D_prime = np.asarray(D_prime)
    shape_factor = np.empty_like(D_prime)

    mask = D_prime > 1.13
    sqrt_term = np.sqrt(D_prime[mask]**2 - 1)

    shape_factor[mask] = (
        (D_prime[mask] - sqrt_term * (1 + 1 / (2 * D_prime[mask]**2)) +
         (1 / D_prime[mask] - 1 / (4 * D_prime[mask]**3)))
        * np.log((D_prime[mask] + sqrt_term) / (D_prime[mask] - sqrt_term))
    )

    shape_factor[~mask] = D_prime[~mask]
    return shape_factor

E_keV_list = np.array([10, 20, 30, 40])
l_in_m_list = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_keV_list * 1e3) 
corr_lengths = (l_in_m_list * prop_in_m) / (px_in_um * 1e-6)

D = np.linspace(1e-6, 40e-6, 100)  # in m

D_prime_all = np.array([D / corr for corr in corr_lengths])
mu_d_all = np.array([mu_d_normalisiert(D_prime) for D_prime in D_prime_all])
"""
plt.figure(figsize=(8, 8))
plt.plot(D*1e6, mu_d_all[0,:], label='E = 10 keV', color = 'green')
plt.plot(D*1e6, mu_d_all[1,:], label='E = 20 keV', color = 'red')
plt.plot(D*1e6, mu_d_all[2,:], label='E = 30 keV', color = 'blue')
plt.plot(D*1e6, mu_d_all[3,:], label='E = 40 keV', color = 'orange')
plt.ylabel(r'$\frac{\varepsilon(D)}{\frac{3}{4} f k^2|\Delta n|^2 \zeta}$')
plt.xlabel("D [$\mu$m]")
plt.xlim(2, 30)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('shape_factor_epsilon_cropped.pdf')
"""
#Calculate prefactor 

n_bkg = 1 - samp2d.delta_bkg + 1j * samp2d.mu_bkg_in_1_m/(2*k_in_1_m)
n_sph = 1 - samp2d.delta_sph + 1j * samp2d.mu_sph_in_1_m/(2*k_in_1_m)
delta_n = (n_bkg - n_sph)
pre = 3*np.pi**2/(l_in_m**2)*f_sph* np.abs(delta_n)**2*corr_length



df_lowres_30keV = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_30keV_lowres.csv')
df_highres_30keV = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_30keV_highres.csv')
df_10keV_fixed = pd.read_csv("dark_field_SiO2_Ethanol_diameter/visibility_results_10keV_v2.csv")
df_20keV_fixed = pd.read_csv("dark_field_SiO2_Ethanol_diameter/visibility_results_20keV_v2.csv")
df_10keV_noabsorb= pd.read_csv('dark_field_SiO2_Ethanol_diameter/corrected_visibility_results_10keV.csv')
df_30keV_fixed = pd.read_csv('dark_field_SiO2_Ethanol_diameter/double_corrected_visibility_results_30keV.csv')
df_40keV_fixed = pd.read_csv('dark_field_SiO2_Ethanol_diameter/double_corrected_visibility_results_40keV.csv')
df_10keV_thin_slices = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_10keV_1mm_slc=0.7um.csv')
df_10keV_thick_slices = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_10keV_1mm_slc=2.8um.csv')
df_10keV_chunky_slices = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_10keV_1mm_slc=15um.csv')

#df_20kev_double_res = pd.read_csv('visibility_results_20keV_double_res.csv')
#df_20keV_tenth_res = pd.read_csv('visibility_results_20keV_tenth_res.csv')

#df_20keV_5_10_7_res = pd.read_csv('visibility_results_20keV_5_10-7_res.csv')

#df_10keV_10_6_res = pd.read_csv('visibility_results_10keV_10-6_res.csv')
#df_20keV_10_6_res = pd.read_csv('visibility_results_20keV_10-6_res.csv')
#df_30keV_10_6_res = pd.read_csv('visibility_results_30keV_10-6_res.csv')
#df_40keV_10_6_res = pd.read_csv('visibility_results_40keV_10-6_res.csv')

df_20keV_01 = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_20keV_1mm_f=0.1.csv')
df_20keV_005 = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_20keV_1mm_f=0.05.csv')

df_20keV_05mm = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_20keV_0.5mm_v2.csv')
df_20keV_2mm = pd.read_csv('dark_field_SiO2_Ethanol_diameter/visibility_results_20keV_2mm_f=0.01.csv')

# Data for correlation plots
df_20keV_corrlength = pd.read_csv('dark_field_parde_correlation_change/visibility_results_20keV_corr_lengths.csv')
df_20keV_corrlength_v2 = pd.read_csv('dark_field_parde_correlation_change/visibility_results_20keV_corr_lengths_v2.csv')
df_20keV_corrlength_20um = pd.read_csv('dark_field_parde_correlation_change/visibility_results_20keV_corr_lengths_20um.csv')
df_40keV_corrlength = pd.read_csv('dark_field_parde_correlation_change/visibility_results_40keV_corr_lengths.csv')


prade_45keV_corrlength = pd.read_csv('dark_field_parde_correlation_change/Prade_remake2_results_45keV_corr_lengths.csv')
prade_45keV_corrlength_v2 = pd.read_csv('dark_field_parde_correlation_change/Prade_remake_10um_results_45keV_corr_lengths_v2.csv')
prade_45keV_corrlength_0966 = pd.read_csv('dark_field_parde_correlation_change/Prade_remake_10um_results_45keV_corr_lengths_0966diam.csv')
prade_45keV_corrlength_0966_05mm = pd.read_csv('dark_field_parde_correlation_change/Prade_remake_10um_results_45keV_corr_lengths_0966diam_05mm.csv')

# Data extracted from prade plot directly
prade_experiment_45keV_corrlength_279 = pd.read_csv('dark_field_parde_correlation_change/prade_data_279.csv')
prade_experiment_45keV_corrlength_0966 = pd.read_csv('dark_field_parde_correlation_change/prade_data_0996.csv')

# skull tests
like_279_bones = pd.read_csv('spheres_bone_air.csv')

#Get prefactors from file for eplsion correlation plot 
df = pd.read_csv("scaled_up_epsilons_v2.csv")
prefactors = df["prefactor"]
wavelength_20keV = df["wavelength"][3]
wavelength_40keV = df["wavelength"][9]
delta_n_squared_20keV = df["delta_n_squared"][3]
delta_n_squared_40keV = df["delta_n_squared"][9]
pre_ohne_corr_20keV = 3*np.pi**2/(wavelength_20keV**2)*f_sph* delta_n_squared_20keV
pre_ohne_corr_40keV = 3*np.pi**2/(wavelength_40keV**2)*f_sph* delta_n_squared_40keV

temp_pre_ohne_corr = 3*np.pi**2/(l_in_m**2)*f_sph* np.abs(delta_n)**2

# correlation lengths from saved simulation data
corr_lengths_20keV = np.array(df_20keV_corrlength_v2["Correlation length"])
corr_lengths_40keV = np.array(df_40keV_corrlength["Correlation length"])
corr_lengths_prade = np.array(prade_45keV_corrlength_v2["Correlation length"])

# Calculating analytical epsilon/ visibility values
D_prime_corr_20keV = d_sph_in_um*1e-6 / corr_lengths_20keV
D_prime_corr_40keV = d_sph_in_um*1e-6 / corr_lengths_40keV
D_prime_corr_prade = d_sph_in_um*1e-6 / corr_lengths_prade

epsilon_20keV = mu_d_normalisiert(D_prime_corr_20keV)*corr_lengths_20keV*pre_ohne_corr_20keV
epsilon_40keV = mu_d_normalisiert(D_prime_corr_40keV)*corr_lengths_40keV*pre_ohne_corr_40keV
epsilon_prade = mu_d_normalisiert(D_prime_corr_prade)*corr_lengths_prade*temp_pre_ohne_corr

V_20keV = np.exp(-epsilon_20keV * t_samp_in_mm * 1e-3)
V_40keV = np.exp(-epsilon_40keV * t_samp_in_mm * 1e-3)
V_prade = np.exp(-epsilon_prade * t_samp_in_mm * 1e-3)
V_prade_scaled = np.exp(-epsilon_prade * 1e-2)
#np.save("Lynch_prade_analyt.npy", V_prade_scaled)
V_prade_scaled_sim = np.exp(-prade_45keV_corrlength_0966['Epsilon'] * 1e-2)
V_prade_scaled_sim_05mm = np.exp(-prade_45keV_corrlength_0966_05mm['Epsilon'] * 1e-2)

df = pd.DataFrame({
    "corr_length_um": corr_lengths_prade * 1e6,
    "V_prade_scaled": V_prade_scaled
})

df.to_csv("prade_analytical_visibility.csv", index=False)

# Plot
plt.figure(figsize=(8, 8))
#plt.plot(D*1e6, mu_d_all[0,:]*prefactors[0], label='E = 10 keV analytical', color='black')
#plt.plot(df_10keV_thin_slices['Sphere size (um)'], df_10keV_thin_slices['Epsilon'], marker='s', linestyle='-', label = r'simulated 0.7 $\mu$m slices', color = 'green')
#plt.plot(df_10keV_fixed['Sphere size (um)'], df_10keV_fixed['Epsilon'], marker='o', linestyle='--', label = r'simulated 1.4 $\mu$m slices', color = 'blue')
#plt.plot(df_10keV_thick_slices['Sphere size (um)'], df_10keV_thick_slices['Epsilon'], marker='^', linestyle='-.', label = r'simulated 2.8 $\mu$m slices', color = 'red')
#plt.plot(df_10keV_chunky_slices['Sphere size (um)'], df_10keV_chunky_slices['Epsilon'], marker='d', linestyle=':', label = r'simulated 15 $\mu$m slices', color = 'orange')
#plt.plot(df_10keV_fixed['Sphere size (um)'], df_10keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 10 keV simulated', color = 'green')
#plt.plot(df_10keV_noabsorb['Sphere size (um)'], df_10keV_noabsorb['Epsilon'], marker='o', linestyle='-', label = 'E = 10 keV simulated half fix', color = 'red')
#plt.plot(D*1e6, mu_d_all[1,:]*prefactors[3], label='analytical', color='black')

#plt.plot(df_20keV_05mm['Sphere size (um)'], df_20keV_05mm['Epsilon'], marker='o', linestyle='-', label = 'simulated total sample thickness 0.5 mm', color = 'red')
#plt.plot(df_20kev_double_res['Sphere size (um)'], df_20kev_double_res['Epsilon'], marker='o', linestyle='-', label = r'simulated $5x10^{-9}$ m pixel size', color = 'orange')
#plt.plot(df_20keV_fixed['Sphere size (um)'], df_20keV_fixed['Epsilon'], marker='o', linestyle='-', label = r'simulated $1x10^{-8}$ m pixel size', color = 'blue')
#plt.plot(df_20keV_tenth_res['Sphere size (um)'], df_20keV_tenth_res['Epsilon'], marker='o', linestyle='-', label = r'simulated $1x10^{-7}$ m pixel size', color = 'green')
#plt.plot(df_20keV_5_10_7_res['Sphere size (um)'], df_20keV_5_10_7_res['Epsilon'], marker='o', linestyle='-', label = r'simulated $5x10^{-7}$ m pixel size', color = 'purple')
#plt.plot(df_20keV_10_6_res['Sphere size (um)'], df_20keV_10_6_res['Epsilon'], marker='o', linestyle='-', label = r'simulated $1x10^{-6}$ m pixel size', color = 'red')
#plt.plot(df_20keV_2mm['Sphere size (um)'], df_20keV_2mm['Epsilon'], marker='o', linestyle='-', label = 'simulated total sample thickness 2 mm', color = 'green')
#plt.plot(D*1e6, mu_d_all[1,:]*prefactors[4], label='analytical particle fraction 5%', color='magenta')
#plt.plot(df_20keV_005['Sphere size (um)'], df_20keV_005['Epsilon'], marker='o', linestyle='-', label = 'simulated particle fraction 5%', color = 'magenta')
#plt.plot(D*1e6, mu_d_all[1,:]*prefactors[5], label='analytical particle fraction 10%', color='cyan')
#plt.plot(df_20keV_01['Sphere size (um)'], df_20keV_01['Epsilon'], marker='o', linestyle='-', label = 'simulated particle fraction 10%', color = 'cyan')
#plt.plot(df_20keV['Sphere size (um)'], df_20keV['Epsilon'], marker='o', linestyle='-', label = 'E = 20 keV simulated', color = 'red')

#plt.plot(df_30keV_fixed['Sphere size (um)'], df_30keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated', color = 'blue')
#plt.plot(df_highres_30keV['Sphere size (um)'], df_highres_30keV['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated high res', color = 'blue')
#plt.plot(df_30keV['Sphere size (um)'], df_30keV['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated', color = 'red')


#plt.plot(df_40keV_fixed['Sphere size (um)'], df_40keV_fixed['Epsilon'], marker='o', linestyle='-', label = r'simulated $1x10^{-8}$ m pixel size', color = 'green')
#plt.plot(D*1e6, mu_d_all[0,:]*prefactors[0], label='E = 10 keV analytical', color='green')
#plt.plot(df_10keV_10_6_res['Sphere size (um)'], df_10keV_10_6_res['Epsilon'], marker='o', linestyle='-', label = 'E = 10 keV simulated', color = 'green')
#plt.plot(D*1e6, mu_d_all[1,:]*prefactors[3], label='E = 20 keV analytical', color='red')
#plt.plot(df_20keV_fixed['Sphere size (um)'], df_20keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 20 keV simulated', color = 'red')
#plt.plot(D*1e6, mu_d_all[2,:]*prefactors[6], label='E = 30 keV analytical', color = 'blue')
#plt.plot(df_30keV_fixed['Sphere size (um)'], df_30keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 30 keV simulated', color = 'blue')
#plt.plot(D*1e6, mu_d_all[3,:]*prefactors[9], label='E = 40 keV analytical', color = 'orange')
#plt.plot(df_40keV_fixed['Sphere size (um)'], df_40keV_fixed['Epsilon'], marker='o', linestyle='-', label = 'E = 40 keV simulated', color = 'orange')
plt.plot(corr_lengths_prade*1e6, V_prade, label='analytical', color = 'blue')
#plt.plot(prade_45keV_corrlength_v2['Correlation length']*1e6, V_prade_scaled_sim_05mm, marker='o', linestyle='-', label = 'simulated', color = 'red')
plt.scatter(like_279_bones['Correlation length']*1e6, like_279_bones['Visibility'], label='experimental', color = 'green')
#plt.plot(corr_lengths_40keV*1e6, V_40keV, label='E = 40 keV analytical', color = 'orange')
#plt.plot(df_40keV_corrlength['Correlation length']*1e6, df_40keV_corrlength['Visibility'], marker='o', linestyle='-', label = 'E = 40 keV simulated', color = 'orange')
#plt.plot(df_20keV_corrlength_20um['Correlation length']*1e6, df_20keV_corrlength_20um['Visibility'], marker='o', linestyle='-', label = 'E = 20 keV simulated 20 um sphere diameter', color = 'blue')
#plt.plot(corr_lengths_20keV*1e6, V_20keV, label='E = 20 keV analytical', color = 'blue')
# Labels and title
#plt.xlabel(r'Correlation length in $\mu$m', fontsize=16)
plt.xlabel(r'Sphere diameter in $\mu$m', fontsize=16)
plt.ylabel('Epsilon', fontsize=16)

#plt.title('Epsilon vs Sphere Diameter at 10 keV')
plt.title(
    f"Epsilon vs Sphere Diameter at resolution $1\\times10^{{-8}}$ m\n"
    f"Thickness of sample: {t_samp_in_mm:.1f} mm | Particle fraction: {f_sph:.2f}",
    fontsize=16
)
#plt.title(f"Epsilon vs Sphere Diameter at at resolution $1x10^{-6}$ \n Thickness of sample: {t_samp_in_mm:.1f} mm | Particle fraction: {f_sph:.2f}", fontsize=16)
#plt.title(f"Epsilon vs Correlation length at {E_in_keV:.1f} keV & particle size {d_sph_in_um} um \n simulated at 0.5 mm & scaled to 1 cm | Particle fraction: {f_sph:.3f}", fontsize=16)
#plt.xlim(0, 2.5)
#plt.xlim(0, 30)
#plt.ylim(0.25, 1.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=12, loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("test_comp232.pdf", dpi=300, bbox_inches='tight')


def G(zeta):
    term1 = np.sqrt(1 - (zeta / 2)**2) * (1 + (1/8) * zeta**2)
    term2 = 0.5 * zeta**2 * (1 - (zeta / 4)**2) * np.log(zeta / (2 + np.sqrt(4 - zeta**2)))
    return term1 + term2

def xray_sld(total_electrons, molar_mass, density):
    """
    Calculate X-ray Scattering Length Density (SLD) in Å⁻²
    Inputs:
        total_electrons: total Z per molecule
        molar_mass: in g/mol
        density: in g/cm³
    """
    N_A = 6.022e23
    V_mol = (molar_mass / density) / N_A 
    return (r_e * total_electrons) / V_mol

zeta10 = corr_lengths[0]/(D/2)
zeta20 = corr_lengths[1]/(D/2)

sld_sph = xray_sld(total_electrons = 30, molar_mass = 60.08, density = samp2d.rho_sph_in_g_cm3 )
sld_bkg = xray_sld(total_electrons = 20, molar_mass = 46.07, density = samp2d.rho_bkg_in_g_cm3 )
delta_sld = sld_sph - sld_bkg
sigmas = 3/2 * l_in_m**2 * D/2 *f_sph * delta_sld**2


"""
new_data = {
    "energy": [E_in_keV],   
    "particle fraction": [f_sph],
    "wavelength": [l_in_m],
    "corr_length": [corr_length],
    "delta_n_squared": [np.abs(delta_n)**2],
    "prefactor": [pre],
}

new_row = pd.DataFrame(new_data)

new_row.to_csv("scaled_up_epsilons_v2.csv", mode='a', header=False, index=False)
"""