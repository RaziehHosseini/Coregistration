# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:46:51 2019

@author: Razieh
"""
import numpy as np
import scipy.io as spio    # 1.0.0
import scipy.signal as spsig
from matplotlib import cm
import imageio
from skimage import io
from skimage.feature import plot_matches
import sys
sys.path.append('./')
import sift_min as sift
                                                                               # TODO adding path to directory in read_img
import point_detectors
import match_points

def read_img(data1, key1, data2, key2, path):
    
    # import data
    data1path = path + data1
    data1_dict = spio.loadmat(data1path)
    master = data1_dict[key1]             # master in complex format
    data2path = path + data2
    data2_dict = spio.loadmat(data2path)
    slave = data2_dict[key2]                   # slave in complex format

    # Crop the image
    mstr_amp = np.absolute( master )
    mstr_amp_se = mstr_amp[835:1618, 5922:10452]
    slv_amp =  np.absolute( slave )
    slv_amp_se = slv_amp [835:1618, 5922:10452] 
    
    # Remove -inf and clip the histogram
    mstr_amp_se[mstr_amp_se<0] = 0
    mstr_amp_se = np.clip(mstr_amp_se, 0, 2)
    slv_amp_se[slv_amp_se<0] = 0
    slv_amp_se = np.clip(slv_amp_se, 0, 2)
    
    # Save image as tiff format
    imageio.imsave(path + 'master.tiff', mstr_amp_se)
    imageio.imsave(path + 'slave.tiff', slv_amp_se)
    
    return mstr_amp_se, slv_amp_se

def read_img_nocrop(data1, key1, data2, key2, path):

    # import data
    data1path = path + data1
    data1_dict = spio.loadmat(data1path)
    master = data1_dict[key1]             # master in complex format
    data2path = path + data2
    data2_dict = spio.loadmat(data2path)
    slave = data2_dict[key2]                   # slave in complex format

    mstr_amp_se = np.absolute( master )
    slv_amp_se = np.absolute( slave )
    
    # Remove -inf and clip the histogram
    mstr_amp_se[mstr_amp_se<0] = 0
    mstr_amp_se = np.clip(mstr_amp_se, 0, 2)
    slv_amp_se[slv_amp_se<0] = 0
    slv_amp_se = np.clip(slv_amp_se, 0, 2)
    
    # Save image as tiff format    
    imageio.imsave(path +'master.tiff', mstr_amp_se)
    imageio.imsave(path +'slave.tiff', slv_amp_se)
    
    return mstr_amp_se, slv_amp_se

def read_img_nocrop_opt(data1, key1, data2, key2, path):

    # import data
    data1path = path + data1
    data1_dict = spio.loadmat(data1path)
    master = data1_dict[key1]             # master in complex format
    data2path = path + data2
    data2_dict = spio.loadmat(data2path)
    slave = data2_dict[key2]                   # slave in complex format

    mstr_amp_se = np.absolute( master )
    slv_amp_se = np.absolute( slave )
    
#    # Remove -inf and clip the histogram
#    mstr_amp_se[mstr_amp_se<0] = 0
#    mstr_amp_se = np.clip(mstr_amp_se, 0, 2)
#    slv_amp_se[slv_amp_se<0] = 0
#    slv_amp_se = np.clip(slv_amp_se, 0, 2)
    
    # Save image as tiff format    
    imageio.imsave(path +'master.tiff', mstr_amp_se)
    imageio.imsave(path +'slave.tiff', slv_amp_se)
    
    return mstr_amp_se, slv_amp_se