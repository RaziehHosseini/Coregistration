# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:52:47 2019

@author: p-sem-2019
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import read_Img as ri
from PIL import Image
import imageio
import IPython.display



def display(mstr_amp, slv_amp):
    ####################
    ### show images ####
    ####################
    ### Master
    # plot with vertical colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mstr_amp, interpolation='nearest', cmap=cm.gray)
    ax.set_title('Intensities of master in [dB]')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, mstr_amp.max()])
    plt.axis('tight')
    
    #### Slave
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(slv_amp, interpolation='nearest', cmap=cm.gray)
    ax.set_title('Intensities of slave in [dB]')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, slv_amp.max()])
    plt.axis('tight')
    plt.show()

# following function is taken from Image Analysis I
def imshow3D(*I):
    """Shows the array representation of one or more images in a jupyter notebook.

        Parameters
        ----------
        I : ndarray of float64
            Array representation of an image
            Concatenates multiple images

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """

    if len(I) == 1:
        I = I[0]
    else:
        channels = [i.shape[2] for i in I]
        heights = [i.shape[0] for i in I]
        max_height = max(heights)
        max_channels = max(channels)

        if min(channels) != max_channels:  # if one image has three channels ..
            I = list(I)
            for i in range(len(I)):
                dim = channels[i]
                if dim == 1:  # .. and another has one channel ..
                    I[i] = np.dstack((I[i], I[i], I[i]))  # .. expand that image to three channels!

        if min(heights) != max_height:  # if heights of some images differ ..
            I = list(I)
            for i in range(len(I)):
                h, w, d = I[i].shape
                if h < max_height:  # .. expand by 'white' rows!
                    I_expanded = np.ones((max_height, w, d), dtype=np.float64) * 255
                    I_expanded[:h, :, :] = I[i]
                    I[i] = I_expanded

        seperator = np.ones((max_height, 3, max_channels), dtype=np.float64) * 255
        seperator[:, 1, :] *= 0
        I_sep = []
        for i in range(len(I)):
            I_sep.append(I[i])
            if i < (len(I) - 1):
                I_sep.append(seperator)
        I = np.hstack(I_sep)  # stack all images horizontally

    assert I.ndim == 3
    h, w, d = I.shape
    assert d in {1, 3}
    if d == 1:
        I = I.reshape(h, w)
    IPython.display.display(Image.fromarray(I.astype(np.ubyte)))



#img1 , img2 = ri.read_img('Master.mat','Master','Slave.mat','Slave')
img1 , img2 = ri.read_img('master1.mat','master','slave1.mat','slave')

#img = cv2.imread(img1,0)
img = np.array(img1)
gray = Image.fromarray(img1)

h, w = img.shape
img1 = img1.reshape((h, w, 1)).astype(np.float64)
imshow3D(img1)