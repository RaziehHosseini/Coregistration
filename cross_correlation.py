# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:11:40 2019

@author: Razieh
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage import io          # 0.15.0
import os.path 
                                                                              
import point_detectors
import match_points
import point_matching
import read_data
import display
import sift_min as sift

if __name__=="__main__":
    
    PATH = "./planet/"
    MASTER_MAT = 'master.mat'
    KEY1 = 'master'
    SLAVE_MAT = 'slave.mat'
    KEY2 = 'slave'
    
    # To check if tiff image exist 
    is_master = os.path.exists(PATH+'master.tiff')
    is_slave = os.path.exists(PATH+'slave.tiff')
    if (is_master == False) and (is_slave == False):
        [mstr_amp, slv_amp] = read_data.read_img_nocrop_opt(MASTER_MAT, KEY1, SLAVE_MAT, KEY2, PATH)
    
    # Read tiff image as input
    my_img_master = io.imread(PATH + 'master.tiff').astype(np.float32)
    my_img_slave = io.imread(PATH + 'slave.tiff').astype(np.float32)
    

shift = (-22.4, 13.32)
# The shift corresponds to the pixel offset relative to the reference image
offset_image = fourier_shift(np.fft.fftn(image), shift)
offset_image = np.fft.ifftn(offset_image)
print("Known offset (y, x): {}".format(shift))

# pixel precision first
shift, error, diffphase = register_translation(image, offset_image)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

plt.show()

print("Detected pixel offset (y, x): {}".format(shift))

# subpixel precision
shift, error, diffphase = register_translation(image, offset_image, 100)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Calculate the upsampled DFT, again to show what the algorithm is doing
# behind the scenes.  Constants correspond to calculated values in routine.
# See source code for details.
cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Supersampled XC sub-area")


plt.show()

print("Detected subpixel offset (y, x): {}".format(shift))