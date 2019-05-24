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
import imageio
from skimage import io
                                                                               # TODO adding path to directory in read_img
                                                                               # working with relative directory (to add module from different directory)
import sys
sys.path.append('./bau')
import main_backbone_bau

import point_detectors
import match_points

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
    
    imageio.imsave('./kerman_data/master.tiff', mstr_amp_se)
    imageio.imsave('./kerman_data/slave.tiff', slv_amp_se)
    
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
    ax = fig.add_subplot(111)
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
#    MASTER_FILE = './noiseless_data/master.mat'
#    KEY1 = 'master0'
#    SLAVE_FILE = './noiseless_data/slave.mat'
#    KEY2 = 'slave0'
    
    # Kerman data
    MASTER_FILE = './kerman_data/master_crop.mat'
    KEY1 = 'master_crop'
    SLAVE_FILE = './kerman_data/slave_crop.mat'
    KEY2 = 'slave_crop'
    
    [mstr_amp, slv_amp] = read_img_nocrop(MASTER_FILE, KEY1, SLAVE_FILE, KEY2)
#    display(mstr_amp, slv_amp)
    
    # Read tiff image as input
    my_img_master = io.imread('./kerman_data/master.tiff')
    my_img_slave = io.imread('./kerman_data/slave.tiff')
    display (my_img_master, my_img_slave)


    """================================ point detection============================
    ==============================================================================="""
#   # Implement harris corner detector
#   master_points,_ = point_detectors.harris_skimage(my_img_master)
#   slave_points,_ = point_detectors.harris_skimage(my_img_slave)
    
    
#    # Implement foerstner detector
    master_points,_ = point_detectors.foerstner_skimage(my_img_master)
    slave_points,_ = point_detectors.foerstner_skimage(my_img_slave)  

   # Implement fast corner detector
#    master_points,_ = point_detectors.fast_skimage(my_img_master)
#    slave_points,_ = point_detectors.fast_skimage(my_img_slave)
#   
	
    # Tiled matching
     #master_points,_ = point_detectors.tiled_point_detection(my_img_master, 6, method = "kitchen_rosenfeld_skimage")
     #slave_points,_ = point_detectors.tiled_point_detection(my_img_slave, 6, method = "kitchen_rosenfeld_skimage")
    
    """================================ Matching ============================
    =============================================================================="""
    matcher = match_points.Matcher(my_img_master,  my_img_slave, master_points, slave_points)
    src, dest = matcher.match()
    
    