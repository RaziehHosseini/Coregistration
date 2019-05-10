# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:21:31 2019

@author: p-sem-2019
"""

#import opencv as cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.draw import ellipse
import sys

#sys.path.append('../imgs')

def harris_skimage(image):
    
    coords = corner_peaks(corner_harris(image), min_distance=1) # larger distance -> fewer points

    coords_subpix = corner_subpix(image, coords, window_size=13)
    
#    print('Number of detected points {}'.format(coords.count()))
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], '.r', markersize=5)
    plt.show()
    

