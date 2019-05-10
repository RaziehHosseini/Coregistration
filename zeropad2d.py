# -*- coding: utf-8 -*-
"""
Taken from summer school edu

function for zeropadding of input spectrum at center for oversampling of dataset in time domain

INPUT: Spectrum (created by, e.g. numpy.fft.fft2(.)) and oversampling factor, e.g., 8
OUTPUT: Zeropadded spectrum. Oversampled data can be obtained by inverse fft, e.g., numpy.fft.ifft2(.)

@author: Stefan Gernhardt

"""

import numpy as np
import sys

def zeropad2d(data, factor):
    rw,cl = data.shape

    # 1st insertion: mid of rows
    data_slice_1 = data[0:np.int(np.ceil(rw/2)),:]
    data_slice_2 = data[np.int(np.floor(rw/2)):,:]
    zeros_mid = np.zeros( ((factor-1)*rw, cl) )
    # insert zeros at center
    data_zp = np.vstack([ data_slice_1, zeros_mid, data_slice_2 ])
    
    # 2nd insertion: mid of columns
    rw2,cl2 = data_zp.shape
    data_slice_1 = data_zp[:,:np.int(np.ceil(cl2/2))]
    data_slice_2 = data_zp[:,np.int(np.floor(cl/2)):]
    zeros_mid = np.zeros( (rw2,(factor-1)*cl2) )
    # insert zeros at center    
    data_zp = np.hstack([ data_slice_1, zeros_mid, data_slice_2 ])
    
    return(data_zp)
    sys.exit(0)