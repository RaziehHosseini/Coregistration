# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:58:22 2019

@author: seyedeh.razieh

=====> to test different matching techniques only change kwargs before matching part
=====> 

"""

import matplotlib.pyplot as plt # 3.1.0
import numpy as np              # 1.16.4
import scipy.io as spio         # 1.3.0
import scipy.signal as spsig
from matplotlib import cm
import imageio                  # 2.5.0
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

def lineno():
    return inspect.currentframe().f_back.f_lineno     

    
if __name__=='__main__':
    
    """================================ Loading Data ============================
    ===============================================================================""" 
#    PATH = "./ori_data/"
#    MASTER_FILE = 'master.mat'
#    KEY1 = 'master'
#    SLAVE_FILE = 'slave.mat'
#    KEY2 = 'slave'
#    
#    PATH = "./noiseless_data/"
#    MASTER_FILE = 'master.mat'
#    KEY1 = 'master0'
#    SLAVE_FILE = 'slave.mat'
#    KEY2 = 'slave0'
    
#    PATH = "./kerman_data/"
#    MASTER_MAT = 'master_crop.mat'
#    KEY1 = 'master_crop'
#    SLAVE_MAT = 'slave_crop.mat'
#    KEY2 = 'slave_crop'
    
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
    
    fig = plt.figure("before stretch")
    plt.hist(np.ravel(my_img_slave), bins=20)
    plt.hist(np.ravel(my_img_master), bins=20)
    
#    display.display (my_img_master, my_img_slave, '0', PATH[2:-1]) # 0 was line number !not good idea
    my_img_master = 255*(my_img_master-np.min(my_img_master))/(np.max(my_img_master) - np.min(my_img_master))#500)#
#    my_img_slave = 255*(my_img_slave-np.min(my_img_slave))/(np.max(my_img_slave) -np.min(my_img_slave)) # 500)# 
#    my_img_master = 255*(my_img_master-920)/(1720 - np.min(my_img_master))
    my_img_slave = 255*(my_img_slave-1000)/(1720 - 1000)
    display.display (my_img_master, my_img_slave, '0', PATH[2:-1]) # 0 line number
    fig = plt.figure("after streching")
    plt.hist(np.ravel(my_img_master), bins=20)

    """================================ Point Detection ============================
    ==============================================================================="""  
   # Implement harris corner detector
#    start_detection_t = time.time()
#    kwargs = {"method":"harris_skimage", "counter":"0"}
#    m_detector = point_detectors.PointDetector(PATH, "master_")
#    master_points,ma_subpixel = m_detector.harris_skimage(my_img_master, num_peaks=500, **kwargs) # selects 500 best points based on cornerness
#    s_detector = point_detectors.PointDetector(PATH, "slave_")
#    slave_points,sl_subpixel = s_detector.harris_skimage(my_img_slave, num_peaks=500, **kwargs)   
#    print("------- detection time: ", time.time()-start_detection_t, '[s]')
#    
    # Implement foerstner detector
    start_detection_t = time.time()
    kwargs = {"method":"foerstner_skimage", "alpha":0.08}
    m_detector = point_detectors.PointDetector(PATH, "master_")
    master_points,ma_subpixel = m_detector.foerstner_skimage(my_img_master, num_peaks= 400, **kwargs)
    s_detector = point_detectors.PointDetector(PATH, "slave_")
    slave_points,sl_subpixel = s_detector.foerstner_skimage(my_img_slave, num_peaks= 400, **kwargs)  
    print("------- detection time: ", time.time()-start_detection_t, '[s]')
  
    # Implement fast corner detector
#    start_detection_t = time.time()
#    kwargs = {"method":"fast_skimage"}
#    m_detector = point_detectors.PointDetector(PATH, "master_")
#    master_points,ma_subpixel = m_detector.fast_skimage(my_img_master, **kwargs)
#    s_detector = point_detectors.PointDetector(PATH, "slave_")
#    slave_points,sl_subpixel = s_detector.fast_skimage(my_img_slave, **kwargs)
#    print("------- detection time: ", time.time()-start_detection_t, '[s]')




	
#    # Tiled matching
#    start_detection_t = time.time()
#    kwargs = {"method":"harris_skimage"}
#    m_detector = point_detectors.PointDetector(PATH, "master_tile_")
#    master_points, ma_subpixel = m_detector.tiled_point_detection(my_img_master, 2, num_peaks= 500, method = "shi_tomasi_skimage")
#    s_detector = point_detectors.PointDetector(PATH, "slave_tile_")
#    slave_points,sl_subpixel = s_detector.tiled_point_detection(my_img_slave, 2, num_peaks= 500, method = "shi_tomasi_skimage")
#    print("------- detection time: ", time.time()-start_detection_t, '[s]')

    """================================ SIFT descriptors ========================
    =============================================================================="""
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
    # Point based matching
    # Extract grey value
#    matching_matrix_ma = point_matching.extract_greyval(master_points, my_img_master)  
#    matching_matrix_sl = point_matching.extract_greyval(slave_points, my_img_slave)
#    
#    # Find the corresponding points and detect inliers
#    cor_points, cor_points_in, cor_sub= point_matching.point_based_matching(matching_matrix_ma, matching_matrix_sl,
#                                                    ma_subpixel, sl_subpixel)
#
#    # Compute the whole flow vector
#    flow_vector = point_matching.get_flow_vec(cor_points)
#    np.savetxt('Flow vector (all)', flow_vector)
#    
#    point_matching.visualization_vec(my_img_master, flow_vector, 'flow_vec')
#    
#    # Compute the flow vector after blunder detection
#    flow_vector_in = point_matching.get_flow_vec(cor_points_in)
#    np.savetxt('Flow vector (inlier)', flow_vector_in)
#    
#    flow_vec_vis = flow_vector_in
#    flow_vec_vis[:,2:4] = flow_vec_vis[:,2:4]*100
#    point_matching.visualization_vec(my_img_master, flow_vec_vis, 'flow_vec_in')
#    
    # Other matching method
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

    display.draw_fv(my_img_master, matches_all_coord2, flow_vec2,
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
    final_flow_small  = final_flow[final_flow_mag<5]
    final_coords_small = final_coords [final_flow_mag<5]
    final_flow_mag_small = np.linalg.norm(final_flow_small, axis=1)
    display.plot_hist(final_flow_mag_small, 'hist of flow_vectors magnitude after 2 directional match (ignoring large vectors)_'+ PATH[2:-1])


    
    display.draw_fv(my_img_master ,final_coords_small, final_flow_small,
                    'bidirectional_after_threshold_on_mag '+kwargs['method']+
                      ' stom'+ PATH[2:-1]+'_matching_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])
    
    """=============================================================================================================================
    ================================             Cross correlation matching            =============================================
    ================================================================================================================================"""
     # from master -> slave
#    kwargs = {"window_size": 21, "descriptor_master":descriptor_master, 
#              "descriptor_slave": descriptor_slave,
#              "method": "cross_correlation_matcher",
#              "similarity": "SSD"}
#    start_detection_t = time.time()
#    matcher = match_points.Matcher(my_img_master,  my_img_slave, master_points, slave_points, similarity=kwargs['similarity'],scorer_beta=0.999)
#    src_master, dest_slave = matcher.match(**kwargs)
#    
#    flow_vec = src_master-dest_slave
#    flow_vec_magnitude = np.linalg.norm(flow_vec, axis=1)
#    matches_all_coord1 = np.hstack((src_master,dest_slave))
#    display.plot_hist(flow_vec_magnitude, 'hist of flow_vectors magnitude from '+kwargs['method']+
#                      ' mtos'+ PATH[2:-1]+'_'+ np.str(kwargs['window_size'])+'_'+kwargs['similarity'])
#    display.plot_point_matches(src_master, dest_slave, my_img_master, my_img_slave)
#    print("------- matching time: ", time.time()-start_detection_t, '[s]')

    