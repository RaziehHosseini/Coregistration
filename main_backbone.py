# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:58:22 2019

@author: seyedeh.razieh
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import scipy.signal as spsig
from matplotlib import cm
from zeropad2d import zeropad2d
import imageio
from skimage import io
import point_detectors


def read_img(data1, key1, data2, key2):

    # import data
    data = spio.loadmat(data1)
    master = data[key1]                  # master in complex format
    data = spio.loadmat(data2)
    slave = data[key2]                   # slave in complex format

    # Crop the image
    mstr_amp = np.absolute( master )
    mstr_amp_se = mstr_amp[835:1618, 5922:10452]
    slv_amp =  np.absolute( slave )
    slv_amp_se = slv_amp [835:1618, 5922:10452] 

#    # Histogram of selected area
#    plt.figure('hist')
#    plt.hist(np.ravel(mstr_amp_se[:200,:200]), bins=100, range=(0.0,2.0))
#    plt.show()
    
    # Remove -inf and clip the histogram
    mstr_amp_se[mstr_amp_se<0] = 0
    mstr_amp_se = np.clip(mstr_amp_se, 0, 2)
    slv_amp_se[slv_amp_se<0] = 0
    slv_amp_se = np.clip(slv_amp_se, 0, 2)
    
    # Save image as tiff format
    imageio.imsave('./ori_data/master.tiff', mstr_amp_se)
    imageio.imsave('./ori_data/slave.tiff', slv_amp_se)
    
    return mstr_amp_se, slv_amp_se
    
def read_img_nocrop(data1, key1, data2, key2):

    # import data
    data = spio.loadmat(data1)
    master = data[key1]                  # master in complex format
    data = spio.loadmat(data2)
    slave = data[key2]                   # slave in complex format

    mstr_amp_se = np.absolute( master )
    slv_amp_se = np.absolute( slave )

#    # Histogram of selected area
#    plt.figure('hist')
#    plt.hist(np.ravel(mstr_amp_se[:200,:200]), bins=100, range=(0.0,2.0))
#    plt.show()
    
    # Remove -inf and clip the histogram
    mstr_amp_se[mstr_amp_se<0] = 0
    mstr_amp_se = np.clip(mstr_amp_se, 0, 2)
    slv_amp_se[slv_amp_se<0] = 0
    slv_amp_se = np.clip(slv_amp_se, 0, 2)
    
    # Save image as tiff format
    imageio.imsave('./noiseless_data/master.tiff', mstr_amp_se)
    imageio.imsave('./noiseless_data/slave.tiff', slv_amp_se)
    
    return mstr_amp_se, slv_amp_se
    
def display(mstr_amp, slv_amp):
    
    ### Master
    fig = plt.figure('master')
    ax = fig.add_subplot(111)
    cax = ax.imshow(mstr_amp, interpolation='nearest', cmap=cm.gray)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, mstr_amp.max()])
    plt.axis('tight')
    
    ### Slave
    fig = plt.figure('slave')
    ax = fig.add_subplot(112)
    cax = ax.imshow(slv_amp, interpolation='nearest', cmap=cm.gray)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, slv_amp.max()])
    plt.axis('tight')
    plt.show()


if __name__=='__main__':
#    MASTER_FILE = './ori_data/master.mat'
#    KEY1 = 'master'
#    SLAVE_FILE = './ori_data/slave.mat'
#    KEY2 = 'slave'
    
    # Plot the matlab data as ouput
    MASTER_FILE = './noiseless_data/master.mat'
    KEY1 = 'master0'
    SLAVE_FILE = './noiseless_data/slave.mat'
    KEY2 = 'slave0'
    
    [mstr_amp, slv_amp] = read_img_nocrop(MASTER_FILE, KEY1, SLAVE_FILE, KEY2)
    display(mstr_amp, slv_amp)
    
    # Read tiff image as input
    my_img_master = io.imread('./noiseless_data/master.tiff')
    my_img_slave = io.imread('./noiseless_data/slave.tiff')
    
#   # Implement harris corner detector
#   point_detectors.harris_skimage(my_img_master)
#   point_detectors.harris_skimage(my_img_slave)
    
    
#    # Implement foerstner detector
#    point_detectors.foerstner_skimage(my_img_master)
#    point_detectors.foerstner_skimage(my_img_slave)  

#   # Implement fast corner detector
#   point_detectors.fast_skimage(my_img_master)
#   point_detectors.fast_skimage(my_img_slave)  
	
    # Harris tiled
    point_detectors.tiled_point_detection(my_img_master, 6, method = "shi_tomasi_skimage")

    
    