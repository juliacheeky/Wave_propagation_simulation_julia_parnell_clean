import numpy as np
import scipy.fft
from skimage.draw import disk
import xraylib as xrl
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


                           
                         
t_samp_in_mm = 1.5
f_sph = 0.15
sim_pix_size_in_m = 1e-6
t_slc_in_pix = int( 0.1e-3 / sim_pix_size_in_m)
d_sph_in_um = 50
img_size_in_pix = 1000
samp_size_in_pix = img_size_in_pix

t_samp_in_pix = int((t_samp_in_mm * 1e-3) / sim_pix_size_in_m)
num_slc = int(t_samp_in_pix /t_slc_in_pix)

d_sph_in_pix = int((d_sph_in_um * 1e-6) / sim_pix_size_in_m)
r_sph_in_pix = int(d_sph_in_pix / 2)                     

num_sph_2dslice = int((samp_size_in_pix * t_samp_in_pix * f_sph)/(np.pi * r_sph_in_pix**2))
print(f'Number of slices: {num_slc}')

    # --- Spheres -------------------------------------------------------------

def draw_sph_centers_2d(seed: int):
    np.random.seed(seed) # set seed to allow reproducability

    # get twice as many a needed, so we can pick the furthest apart
    centres = np.random.randint(r_sph_in_pix, [t_samp_in_pix-r_sph_in_pix, samp_size_in_pix-r_sph_in_pix], (num_sph_2dslice*2, 2))
    # calculate distances^2 between all pairs of points
    distances = np.sum(np.square(centres.reshape((-1, 1, 2)) - centres), -1)
    # ignore values in lower half by setting them to max possible
    distances[np.arange(distances.shape[0])[:,None] >= np.arange(distances.shape[1])] = t_samp_in_pix*samp_size_in_pix
    # get the minumim distance to previous points
    min_distances = np.nanmin(distances, 0)
    # sort indices by descreasing distance, get best half
    indices = np.argsort(-min_distances)[0:num_sph_2dslice]
    return centres.take(indices, 0)


def create_slice2d(seed: int) -> Tuple[np.ndarray, 
                                        np.ndarray]:

    # Create the 2D slice that will contain the projected spheres
    slc2d_sph = np.zeros((samp_size_in_pix,t_samp_in_pix), dtype=np.uint16) 
    
    centres = draw_sph_centers_2d(seed)

    for z,x in centres:
        rr, cc = disk((x,z),r_sph_in_pix, shape=(samp_size_in_pix,t_samp_in_pix))# Create a 2D sphere with the center at (cX, cZ)
        slc2d_sph[rr, cc] = 1


    slc2d_sph_real = np.abs(slc2d_sph)
    
    # Create the background of the slice
    slc2d_bkg = np.ones(slc2d_sph_real.shape) - slc2d_sph_real
    #return slc2d_sph_padded, slc2d_bkg
    return slc2d_sph_real, slc2d_bkg

def create_projected_1d_slices(seed: int) -> Tuple[np.ndarray,np.ndarray]:
    slice_profiles_sph = []
    slice_profiles_bkg = []
    slc2d_sph_real, slc2d_bkg = create_slice2d(seed=0)
    for i in range(num_slc):
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
    
    return slice_profiles_sph, slice_profiles_bkg


slc2d_sph_real, slc2d_bkg = create_slice2d(seed=2)
slice_profiles_sph, slice_profiles_bkg = create_projected_1d_slices(seed=2)
cmap = LinearSegmentedColormap.from_list("sph_bkg", ["#fbdc14","#3c7cfb"])
plt.imshow(slc2d_sph_real, cmap=cmap, vmin=0, vmax=1)
plt.axis('off')
plt.savefig('2d_slice_bkg.pdf', dpi=300, bbox_inches='tight', pad_inches=0)

"""
fig, ax = plt.subplots()

# Expand the image with small gaps between projected slices
gap_px = 50
expanded_width = t_samp_in_pix + (num_slc - 1) * gap_px
# Use NaNs for gaps so they can be rendered white
slc2d_sph_expanded = np.full(
    (samp_size_in_pix, expanded_width),
    np.nan,
    dtype=np.float32,
)

boundary_x = []
for i in range(num_slc):
    src_start = i * t_slc_in_pix
    src_end = src_start + t_slc_in_pix
    dst_start = i * t_slc_in_pix + i * gap_px
    dst_end = dst_start + t_slc_in_pix
    slc2d_sph_expanded[:, dst_start:dst_end] = slc2d_sph_real[:, src_start:src_end]
    if i < num_slc - 1:
        boundary_x.append(dst_end - 0.5 + gap_px / 2.0)

# Use complementary colors for background and spheres, white for gaps
cmap = LinearSegmentedColormap.from_list("sph_bkg", ["#fbdc14","#3c7cfb" ])  # blue / yellow
cmap.set_bad("#ffffff")
ax.imshow(slc2d_sph_expanded, cmap=cmap, vmin=0, vmax=1)

# Draw slice boundaries along the thickness axis for projected slices

ax.axis('off')
#fig.savefig('2d_bkg_lines_seperate.pdf', dpi=300, bbox_inches='tight', pad_inches=0)

# --- 1D projected slices (summed over thickness) ---------------------------
slice_fraction = slice_profiles_sph / t_slc_in_pix  # fraction of spheres per pixel row

slice_width = 10
gap_1d_px = gap_px
compressed_width = num_slc * slice_width + (num_slc - 1) * gap_1d_px
slice_1d_expanded = np.full((samp_size_in_pix, compressed_width), np.nan, dtype=np.float32)

for i in range(num_slc):
    dst_start = i * (slice_width + gap_1d_px)
    dst_end = dst_start + slice_width
    slice_1d_expanded[:, dst_start:dst_end] = slice_fraction[i][:, None]

fig1d, ax1d = plt.subplots()
ax1d.imshow(slice_1d_expanded, cmap=cmap, vmin=0, vmax=1)
ax1d.axis('off')
fig1d.savefig('1d_bkg_profiles.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
"""