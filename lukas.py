# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:07:51 2019

@author: Razieh
"""

import matplotlib.pyplot as plt # 3.1.0
import numpy as np              # 1.16.4
from matplotlib import cm
from PIL import Image
import cv2
import sys
import os
from skimage import io          # 0.15.0


import display
import read_data

sys.path.append('./noiseless_data/')

if __name__=="__main__":
    
#    PATH = r"./planet/"
#    MASTER_MAT = 'master.mat'
#    KEY1 = 'master'
#    SLAVE_MAT = 'slave.mat'
#    KEY2 = 'slave'

    PATH = r"./kerman_data/"
    MASTER_MAT = 'master_crop.mat'
    KEY1 = 'master_crop'
    SLAVE_MAT = 'slave_crop.mat'
    KEY2 = 'slave_crop'
    
    # To check if tiff image exist 
    is_master = os.path.exists(PATH+'master.tiff')
    is_slave = os.path.exists(PATH+'slave.tiff')
    if (is_master == False) and (is_slave == False):
        [mstr_amp, slv_amp] = read_data.read_img_nocrop_opt(MASTER_MAT, KEY1, SLAVE_MAT, KEY2, PATH)    


    my_img_master = io.imread(PATH + 'master.tiff').astype(np.float32)
    my_img_slave = io.imread(PATH + 'slave.tiff').astype(np.float32)
    
    my_img_master = 255*(my_img_master-np.min(my_img_master))/(np.max(my_img_master) - np.min(my_img_master))#500)#
    my_img_slave = 255*(my_img_slave-np.min(my_img_slave))/(np.max(my_img_slave) -np.min(my_img_slave)) # 500)# 

    # 8 bit image, maxCorners,
    # qualityLevel – Minimum accepted quality of image corners,
    # minDistance – Minimum possible Euclidean distance between the returned corners.
    corners = cv2.goodFeaturesToTrack(my_img_master,5000,0.01,10) 
    
    win_size = (105, 105)
    max_level = 1
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    # Perform LK-tracking
    lk_params = {"winSize": win_size,
                 "maxLevel": max_level,
                 "criteria": criteria}
    kp2, st, err = cv2.calcOpticalFlowPyrLK(my_img_master.astype(np.uint8),
                                            my_img_slave.astype(np.uint8),
                                            corners,
                                            None,
                                            **lk_params)
    kp3, st, err = cv2.calcOpticalFlowPyrLK(my_img_slave.astype(np.uint8),
                                            my_img_master.astype(np.uint8),
                                            kp2,
                                            None,
                                            **lk_params)
    
    corners = corners.reshape(-1,2)
    kp3 = kp3.reshape(-1, 2)
    flow_vec = corners-kp3
    final_flow_mag = np.linalg.norm(flow_vec, axis=1)   
    good = final_flow_mag < 1    # keep flow vectors less than 1 pixel magnitude
    corners=corners[good]
    kp2=kp2[good]
    kp3=kp3[good]
    flow_vec = corners-kp3
    final_flow_mag = np.linalg.norm(flow_vec, axis=1)   

    display.draw_fv(my_img_slave, corners, flow_vec,
                    'quiver '+
                      'shitomasi'+ PATH[2:-1]+'_LK_opticalflow_'+ np.str(lk_params['winSize']))
    
#    display.plot_hist(final_flow_mag, 
#                      'hist of flow_vectors mag_shitomasi_LK_opticalflow_distance10_'
#                      + PATH[2:-1], labels="window"+str(win_size))

    # save flow vectors and coordinates in text file or csv   
#    fname= "./fv_save/lukas_subpix_winsize_"+str(win_size)+"_"+PATH[2:-1]+".csv"
#    np.savetxt(fname, np.transpose((corners[:,0],corners[:,1], flow_vec[:,0],flow_vec[:,1])), delimiter=",")

    
    
    