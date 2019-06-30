# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:27:41 2019

@author: Razieh
"""

import matplotlib.pyplot as plt # 3.1.0
import numpy as np              # 1.16.4
import scipy.io as spio         # 1.3.0
import scipy.signal as spsig
from matplotlib import cm
from skimage import io          # 0.15.0
from skimage.feature import plot_matches
import sys
sys.path.append('./')
import os.path                                                                               
import time
import inspect
                                                                             
import point_detectors
import match_points
import point_matching
import read_data
import display
import sift_min as sift
from scipy.ndimage import fourier_shift
from skimage import data


if __name__=='__main__':

    
    my_img_master = data.camera()
    shift = (-20, 20)
    
    # The shift corresponds to the pixel offset relative to the master image
    my_img_slave = fourier_shift(np.fft.fftn(my_img_master), shift)
    my_img_slave = np.fft.ifftn(my_img_slave)
    my_img_slave = my_img_slave.real 

    """=============        Point detection                   =================
    ==========================================================================="""
    PATH = "./testapproach/"
    start_detection_t = time.time()
    kwargs = {"method":"foerstner_skimage","alpha":0.8}
    m_detector = point_detectors.PointDetector(PATH, "master_")
    master_points,ma_subpixel = m_detector.foerstner_skimage(my_img_master, num_peaks= 40, **kwargs)
    s_detector = point_detectors.PointDetector(PATH, "slave_")
    slave_points,sl_subpixel = s_detector.foerstner_skimage(my_img_slave, num_peaks= 40, **kwargs)  
    print("------- detection time: ", time.time()-start_detection_t, '[s]')


    """=============        SIFT descriptors                   =================
    ==========================================================================="""
    descriptor_master=[] 
    descriptor_slave = []
    start_feature_ext_t = time.time()
    descriptor_master, ori_master = sift.sift_descriptor(my_img_master, master_points)
    descriptor_slave, ori_slave = sift.sift_descriptor(my_img_slave, slave_points)
    print("------- SIFT extraction time: ", time.time()-start_detection_t, '[s]')
    
    # window_size in case of "windowbase method" is window around each point to calculate distance measure
    # window_size in case of "cross_correlation" is cross correlation window to calculate cross_correlation 
    # descriptors only for "match_point_wise_nD" make sense
    kwargs = {"window_size": 55, "descriptor_master":descriptor_master, 
              "descriptor_slave": descriptor_slave,
              "method": "match_point_wise_nD",
              "similarity": "SSD"}
    
    """================================ Matching  ============================
    =============================================================================="""
        # from master -> slave
    start_detection_t = time.time()
    matcher = match_points.Matcher(my_img_master,  my_img_slave, master_points, slave_points, similarity=kwargs['similarity'],scorer_beta=0.999)
    src_master, dest_slave = matcher.match(**kwargs)
    
    flow_vec = src_master-dest_slave
#    flow_vec = flow_vec[flow_vec_magnitude<20]
#    src_master = src_master[flow_vec_magnitude<20]
#    dest_slave = dest_slave[flow_vec_magnitude<20]
    matches_all_coord1 = np.hstack((src_master,dest_slave))
    flow_vec_magnitude = np.linalg.norm(flow_vec, axis=1)
    display.plot_hist(flow_vec_magnitude, 'hist of flow_vectors magnitude from '+kwargs['method']+
                      ' mtos_'+ PATH[2:-1]+'_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])
    print("------- matching time: ", time.time()-start_detection_t, '[s]')

    display.draw_fv(my_img_master, matches_all_coord1, flow_vec,
                    'quiver '+kwargs['method']+
                      ' mtos'+ PATH[2:-1]+'_matching_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])

#    # from slave->master
    matcher = match_points.Matcher(my_img_slave, my_img_master, slave_points, master_points, similarity=kwargs['similarity'], scorer_beta=0.999)
    src_slave, dest_master = matcher.match(**kwargs)
    flow_vec2 = src_slave-dest_master
    flow_vec_magnitude2 = np.linalg.norm(flow_vec, axis=1)
#    flow_vec2 = flow_vec2[flow_vec_magnitude2<20]
#    src_slave = src_slave[flow_vec_magnitude2<20]
#    dest_master = dest_master[flow_vec_magnitude2<20]
    matches_all_coord2 = np.hstack((dest_master, src_slave))
    display.plot_hist(flow_vec_magnitude2, 'hist of flow_vectors magnitude from '+kwargs['method']+
                      'stom_'+ PATH[2:-1]+'_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])
#    display.plot_point_matches(src_master, dest_slave, my_img_master, my_img_slave)

    display.draw_fv(my_img_slave, matches_all_coord2, flow_vec2,
                    'quiver '+kwargs['method']+
                      ' stom'+ PATH[2:-1]+'_matching_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])

    
    # keep only matches that exist in both directions (bidirectional)
    final_coords = np.zeros((0, 4), dtype = np.float)
    for row in matches_all_coord1:
        index = np.where((row == matches_all_coord2).all(1))
        if index[0]:
            print(index[0])
            final_coords = np.row_stack((final_coords, matches_all_coord2[index[0], :]))
            
    # flow_vectors after bidirectional ===> number of flow vectors of course reduced but still might have wrong matches/large offset
    final_flow = np.zeros((final_coords.shape[0], 2), dtype = np.float)
    final_flow[:,0] = (final_coords[:,0] -final_coords[:, 2])
    final_flow[:, 1] = (final_coords[:,1] -final_coords[:, 3])
    final_flow_mag = np.linalg.norm(final_flow, axis=1)
    display.plot_hist(final_flow_mag, 'hist of flow_vectors magnitude after 2 directional match_'+ PATH[2:-1])
    
    # to get rid of large flow vectors just put a threshold of 5
    final_flow_small  = final_flow[final_flow_mag<30]
    final_coords_small = final_coords [final_flow_mag<30]
    final_flow_mag_small = np.linalg.norm(final_flow_small, axis=1)
    display.plot_hist(final_flow_mag_small, 'hist of flow_vectors magnitude after 2 directional match (ignoring large vectors)_'+ PATH[2:-1])


    
    display.draw_fv(my_img_master ,final_coords_small, final_flow_small,
                    'bidirectional_after_threshold_on_mag '+kwargs['method']+
                      ' stom'+ PATH[2:-1]+'_matching_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])
    
    
    
    
    