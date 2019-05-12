# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:21:31 2019

@author: p-sem-2019
"""

#import opencv as cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage import feature as sf
from skimage.draw import ellipse
import sys

sys.path.append('../imgs')

"""=========================      helper      ===============================================
============================================================================================="""
def draw_points(img, p, sp=None): # p:detected points sp:detected points (sub-pixels)
    
    fig, ax = plt.subplots()
    ax.set(title=p.tostring)                                                   # TODO (plot results with title and all in one figure)
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(p[:, 1], p[:, 0], '.r', markersize=2)
    #ax.plot(sp[:, 1], sp[:, 0], '.b', markersize=5)
    plt.show()
    
def partition_img(w,h, image, part_num):
    w_new = np.int(w/part_num)
    h_new = h
    img_partitioned = np.zeros((h_new, w_new, part_num))
    for i in range(part_num):
        img_partitioned[:, :, i] = image[:h_new, 0+i*w_new:(i+1)*w_new]
    return img_partitioned

"""===========================    Point detectors   ========================================
============================================================================================="""    
def harris_skimage(image, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_harris(image), min_distance=1) # larger distance -> fewer points
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size=13) # sub pixel accuracy
    draw_points(image, cornerness_matrix, coords_subpix)
    return cornerness_matrix, coords_subpix

def shi_tomasi_skimage(image, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_shi_tomasi(image), min_distance=1)
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size = 13)
    draw_points(image, cornerness_matrix, coords_subpix)
    return cornerness_matrix, coords_subpix
        
def kitchen_rosenfeld_skimage(image, threshold_abs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_kitchen_rosenfeld(image,mode= 'constant'), 
                                        min_distance=1,
                                        threshold_abs=threshold_abs,
                                        threshold_rel=0.3)
    #coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size = 13)
    print("detected points: ",cornerness_matrix.size)
    draw_points(image, cornerness_matrix)
    return cornerness_matrix, coords_subpix

def fast_skimage(image, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_fast(image, 16), min_distance=1)
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size=13)
    draw_points(image, cornerness_matrix)
    return cornerness_matrix, coords_subpix

def foerstner_skimage(image,**kwargs):
    w, q = sf.corner_foerstner(image)
    q_min = 0.5
    w_min = 0.3
    foerstner = (q > q_min) * (w > w_min) * w
    cornerness_matrix = sf.corner_peaks(foerstner, min_distance=1)
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size=13)
    draw_points(image, cornerness_matrix)
    return cornerness_matrix, coords_subpix
    
dict_func = {"harris_skimage":harris_skimage, 
             "shi_tomasi_skimage": shi_tomasi_skimage, 
             "kitchen_rosenfeld_skimage": kitchen_rosenfeld_skimage,
             "fast_skimage":fast_skimage,
             "foerstner_skimage":foerstner_skimage}
   
def tiled_point_detection(image, partition = 2, method = "harris_skimage"):
    
    assert (partition%2==0), "Even partition is not possible!"
    h, w = np.shape(image)
    part_half = partition/2 # to check scale of w w.r.t. h
    detector_method = dict_func.get(method)
    kwargs = {"threshold_abs": np.min(image)}
    total_cornerness = np.zeros_like(image)                                   # TODO (save points to be shown on whole img)
        img_partitioned = partition_img(w,h, image, partition)
        for i in range(img_partitioned.shape[2]):
            detector_method(img_partitioned[:,:,i], **kwargs)
            
    elif part_half*w < h:
        img_partitioned = partition(h, w, image)
        for i in range(img_partitioned.shape[2]):
            detector_method(img_partitioned[:,:,i], **kwargs)
    else:
        # partition along both w and h                                        # TODO (Partition along both h and w)
        print("end")
        
#if __name__== '__main__':
#    kitchen_rosenfeld_skimage(my_img_master)
#    tiled_point_detection(my_img_master, partition=2, method = "foerstner_skimage")
        
                                                                               # Other TODOs: 
                                                                               #             1)get result for matching (flow vectors)
                                                                               #             2) gettting histogram of tile vs num of points
                                                                               #             3) think how to show 
                                                                               