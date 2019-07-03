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
from skimage import io

sys.path.append('../imgs')

"""=========================      helper      ===============================================
============================================================================================="""
def draw_points(img, p, sp=None): # p:detected points sp:detected points (sub-pixels)
    
    fig, ax = plt.subplots()
    #ax.set(title=p.tostring)                                                  # TODO (plot results with title and all in one figure)
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(p[:, 1], p[:, 0], '.r', markersize=2)
    #ax.plot(sp[:, 1], sp[:, 0], '.b', markersize=5)
    plt.show()

"""===========================    Point detectors   ========================================
============================================================================================="""    
def harris_skimage(image, min_distance, num_peaks, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_harris(image), min_distance=min_distance, num_peaks=num_peaks) # larger distance -> fewer points
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size=13, alpha=0.8) # sub pixel accuracy
    draw_points(image, cornerness_matrix, coords_subpix)
    print("detected points: ",cornerness_matrix.shape[0])
    return cornerness_matrix, coords_subpix

def shi_tomasi_skimage(image, min_distance, num_peaks, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_shi_tomasi(image), min_distance=min_distance, num_peaks=num_peaks)
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size = 13, alpha=0.8)
    draw_points(image, cornerness_matrix, coords_subpix)
    print("detected points: ",cornerness_matrix.shape[0])
    return cornerness_matrix, coords_subpix
        
def kitchen_rosenfeld_skimage(image, threshold_abs_kr, min_distance, num_peaks, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_kitchen_rosenfeld(image, mode='constant'), 
                                        min_distance=min_distance, num_peaks=num_peaks,
                                        threshold_abs=threshold_abs_kr,
                                        threshold_rel=0.3)
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size = 13, alpha=0.8)
    print("detected points: ",cornerness_matrix.shape[0])
    draw_points(image, cornerness_matrix)
    return cornerness_matrix, coords_subpix

def fast_skimage(image, min_distance, num_peaks, **kwargs):
    coords_subpix = np.zeros_like(image)
    cornerness_matrix = sf.corner_peaks(sf.corner_fast(image, 16, 0.8), min_distance=min_distance, num_peaks=num_peaks) # no_of_detected_points*2
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size=13, alpha=0.8)
    draw_points(image, cornerness_matrix)
    return cornerness_matrix, coords_subpix

def foerstner_skimage(image, min_distance, num_peaks, **kwargs):
    w, q = sf.corner_foerstner(image)
    q_min = 0.9
    w_min = 0.1
    foerstner = (q > q_min) * (w > w_min) * w
    cornerness_matrix = sf.corner_peaks(foerstner, min_distance=min_distance, num_peaks=num_peaks)
    coords_subpix = sf.corner_subpix(image, cornerness_matrix, window_size=13,alpha=0.8)
    draw_points(image, cornerness_matrix)
    print("detected points: ",cornerness_matrix.shape[0])
    return cornerness_matrix, coords_subpix
"""==========================================================================================
============================================================================================="""   

dict_func = {"harris_skimage":harris_skimage, 
             "shi_tomasi_skimage": shi_tomasi_skimage, 
             "kitchen_rosenfeld_skimage": kitchen_rosenfeld_skimage,
             "fast_skimage":fast_skimage,
             "foerstner_skimage":foerstner_skimage}


def tiled_point_detection(image, partition , method, min_distance=1, num_peaks=50, **kwargs):                    #TODO for every number of partitions(even/odd)-current: only power of two.
    detector_method = dict_func.get(method)
    kwargs = {"threshold_abs_kr": np.min(image),"method": method, 
              "min_distance":min_distance,"num_peaks":num_peaks}
    cornerness_matrix = np.zeros((0,2))
    coords_subpix = np.zeros((0,2))
    h, w = np.shape(image)

    if partition/2!=1:  
        if w < h:
            cornerness_matrix_1, coords_subpix_1 = tiled_point_detection(image[:np.int(h/2), :], partition/2, **kwargs)
            cornerness_matrix_2, coords_subpix_2 = tiled_point_detection(image[np.int(h/2):, :], partition/2, **kwargs)
            cornerness_matrix_2[:,0] = cornerness_matrix_2[:,0] + int(h/2)
            cornerness_matrix = np.row_stack((cornerness_matrix_1, cornerness_matrix_2))
            coords_subpix_2[:,0] = coords_subpix_2[:,0] + float(h/2)
            coords_subpix = np.row_stack((coords_subpix_1, coords_subpix_2))
        else:
            cornerness_matrix_1, coords_subpix_1 = tiled_point_detection(image[: ,:np.int(w/2)], partition/2, **kwargs)
            cornerness_matrix_2, coords_subpix_2 = tiled_point_detection(image[: ,np.int(w/2):], partition/2, **kwargs)
            cornerness_matrix_2[:,1] = cornerness_matrix_2[:,1] + int(w/2) 
            cornerness_matrix = np.row_stack((cornerness_matrix_1, cornerness_matrix_2))
            coords_subpix_2[:,1] = coords_subpix_2[:,1] + float(w/2)
            coords_subpix = np.row_stack((coords_subpix_1, coords_subpix_2))
            
    else:
        if w < h:
            cor_tmp, cor_sub_tmp = detector_method(image[:np.int(h/2), :], **kwargs)
            cornerness_matrix = np.row_stack((cornerness_matrix,cor_tmp))
            coords_subpix = np.row_stack((coords_subpix,cor_sub_tmp))
            
            cor_tmp, cor_sub_tmp = detector_method(image[np.int(h/2):, :], **kwargs)
            cor_tmp[:,0] = cor_tmp[:,0] + int(h/2)
            cornerness_matrix = np.row_stack((cornerness_matrix,cor_tmp))
            cor_sub_tmp[:,0] = cor_sub_tmp[:,0] + float(h/2)
            coords_subpix = np.row_stack((coords_subpix,cor_sub_tmp))
        
        else:
            cor_tmp, cor_sub_tmp = detector_method(image[: ,:np.int(w/2)], **kwargs) 
            cornerness_matrix = np.row_stack((cornerness_matrix,cor_tmp))
            coords_subpix = np.row_stack((coords_subpix,cor_sub_tmp))
            
            cor_tmp, cor_sub_tmp = detector_method(image[: ,np.int(w/2):], **kwargs)
            cor_tmp[:,1] = cor_tmp[:,1] + int(w/2)
            cornerness_matrix = np.row_stack((cornerness_matrix,cor_tmp))
            cor_sub_tmp[:,1] = cor_sub_tmp[:,1] + float(w/2)
            coords_subpix = np.row_stack((coords_subpix,cor_sub_tmp))

    return cornerness_matrix, coords_subpix



if __name__== '__main__':
    my_img_master = io.imread('./kerman_data/master.tiff')
    my_img_slave = io.imread('./kerman_data/slave.tiff')
    
#    kitchen_rosenfeld_skimage(my_img_master, np.my_img_master)
#    master_points = fast_skimage(my_img_master)
#    slave_points = fast_skimage(my_img_slave)
#    matcher = match_points.matcher(my_img_master, my_img_slave, master_points, slave_points)
    
    cornerness_matrix_ma, coords_subpix_ma = tiled_point_detection(
            my_img_master, partition=2, method = "foerstner_skimage", 
            min_distance=5, num_peaks=10)
    
    cornerness_matrix_sl, coords_subpix_sl = tiled_point_detection(
            my_img_slave, partition=2, method = "foerstner_skimage", 
            min_distance=5, num_peaks=10)
    
                                                                               # Other TODOs: 
    
                                                                               #             1)get result for matching (flow vectors)
                                                                               #             2) gettting histogram of tile vs num of points
                                                                               #             3) think how to show 
                                                                               