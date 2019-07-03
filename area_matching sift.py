# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:13:37 2019

@author: Baozzz
"""
import numpy as np
import point_detectors_B
import sift_min as sift
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage.feature import plot_matches

def extract_greyval(cornerness_matrix, image):
    points_num = int(cornerness_matrix.shape[0])
    coord_x = cornerness_matrix[:,0].reshape((points_num,1)).astype('int64')
    coord_y = cornerness_matrix[:,1].reshape((points_num,1)).astype('int64')
    gray_val = image[coord_x,coord_y].reshape((points_num,1))
    matching_matrix = np.hstack((coord_x, coord_y, gray_val))
    
    return matching_matrix
    
def point_based_matching(matrix_ma, matrix_sl, coords_subpix_ma, coords_subpix_sl,
                         weight=0.5, dis_thres = 20):
    # Initialize the variable
    num_ma = matrix_ma.shape[0]
    num_sl = matrix_sl.shape[0]
    candi_1 = np.zeros((0,6))      # Corresponding points candidate from ma --> sl
    candi_2 = np.zeros((0,6))      # Corresponding points candidate from sl --> ma
    candidate = np.zeros((0,6))        # Final corresponding points
    loss = np.zeros((max(num_ma,num_sl),7))
    loss_min = np.zeros((1,7))
    candi_tmp = np.zeros((0,6))
    
    # Compare corresponding points from master to slave
    for k in range(2):
        if k == 1:                                                             # Inverse the comparison
            num_ma, num_sl = num_sl, num_ma
            matrix_ma, matrix_sl = matrix_sl, matrix_ma
        for i in range(num_ma):
            for j in range(num_sl):
                x_lo_diff = np.abs(matrix_ma[i,0]-matrix_sl[j,0])
                y_lo_diff = np.abs(matrix_ma[i,1]-matrix_sl[j,1])
                grey_diff = np.abs(matrix_ma[i,2]-matrix_sl[j,2])
                dis = np.sqrt(x_lo_diff**2 + y_lo_diff**2)
                
                if dis > dis_thres:
                    continue
                
                lo = 1/(np.exp(dis))
#               lo = 1/(np.sqrt(x_lo_diff**2 + y_lo_diff**2)+0.001)                
                loss[j,0] = 1 - (weight*(lo) + (1-weight)*grey_diff)           # Score for points matching
                
                if k == 1:
                    loss[j,3:5] = matrix_ma[i,0:2]                             # Store the location in sl
                    loss[j,1:3] = matrix_sl[j,0:2]                             # Store the location in ma
                    loss[j,5] = j
                    loss[j,6] = i
                else:
                    loss[j,1:3] = matrix_ma[i,0:2]                             # Store in ma
                    loss[j,3:5] = matrix_sl[j,0:2]                             # Store in sl
                    loss[j,5] = i
                    loss[j,6] = j
            
#            loss = loss[~(loss[:,0:7]==0).all(1)]                              # Remove all 0 value
#            loss = loss[loss[:,0].argsort()]                                   # Resort the score value
#            loss_min = loss[0:int(loss.shape[0]/2),:].reshape((int(loss.shape[0]/2),7))                                # Best score for matching
#            
#            if k == 1:                                                         # Cor-points from sl-->ma
#                candi_tmp = loss_min[0:,1:7].reshape((int(loss.shape[0]/2),6))
#                candi_2 = np.row_stack((candi_2,candi_tmp))
#                candi_2 = candi_2[~(candi_2[:,0:4]==0).all(1)]         # Remove all 0 rows
#            else:                                                              # Cor-points from ma-->sl
#                candi_tmp = loss_min[0:,1:7].reshape((int(loss.shape[0]/2),6))
#                candi_1 = np.row_stack((candi_1,candi_tmp))
#                candi_1 = candi_1[~(candi_1[:,0:4]==0).all(1)]
#            loss = np.zeros((max(num_ma,num_sl),7))                        # Initialize the score
#            
            loss = loss[~(loss[:,0:7]==0).all(1)]                              # Remove all 0 value
            loss = loss[loss[:,0].argsort()]                                   # Resort the score value
            candi_tmp = loss[:,1:7].reshape((int(loss.shape[0]),6))
            
            if k == 1:                                                         # Cor-points from sl-->ma
                candi_2 = np.row_stack((candi_2,candi_tmp))
                candi_2 = candi_2[~(candi_2[:,0:4]==0).all(1)]         # Remove all 0 rows
            else:                                                              # Cor-points from ma-->sl
                candi_1 = np.row_stack((candi_1,candi_tmp))
                candi_1 = candi_1[~(candi_1[:,0:4]==0).all(1)]
            loss = np.zeros((max(num_ma,num_sl),7))                        # Initialize the score
    
    # Select detected common corresponding pair
    for i in range(candi_1.shape[0]):
        for j in range(candi_2.shape[0]):
            if (candi_1[i] == candi_2[j]).all:
                candidate = np.row_stack((candidate,candi_1[i]))
                break
            
    candidate = np.unique(candidate, axis=0)                                 # Remove repeated detection points
       
    # Get subpixel accuracy
    candi_sub_tmp = np.zeros((1,4))
    candi_sub = np.zeros((0,4))
    for i in range(candidate.shape[0]):
        candi_sub_tmp[:,0:2] = coords_subpix_ma[int(candidate[i,4]),:]
        candi_sub_tmp[:,2:4] = coords_subpix_sl[int(candidate[i,5]),:]
        candi_sub = np.row_stack((candi_sub, candi_sub_tmp))
    
    candi_sub = candi_sub[~np.isnan(candi_sub).any(1)]

    return candidate, candi_sub

def sift_matching(descriptor_master, descriptor_slave, candi_sub):
    num_ma = descriptor_master.shape[0]
    num_sl = descriptor_slave.shape[0]
    cor_points_sift_ma = np.zeros((1,2))
    cor_points_sift_sl = np.zeros((1,2))
    cor_points_sift_1 = np.zeros((0,4))
    cor_points_sift_2 = np.zeros((0,4))
    cor_points_sift = np.zeros((0,4))
            
    for k in range(2):
        if k == 1:                                                             # Inverse the comparison
            num_ma, num_sl = num_sl, num_ma
            descriptor_master, descriptor_slave = descriptor_slave, descriptor_master
                    
        cor_points = np.zeros((0,2))
        for i in range(num_ma): 
            sift_diff = np.zeros((max(num_ma,num_sl),3))
            for j in range(num_sl):
#                sift_diff_sum = 0
                sift_diff_sum_1 = 0
                sift_diff_sum_2 = 0
                sift_diff_sum_3 = 0
                if np.sqrt((candi_sub[i,0]-candi_sub[j,2])**2 +(candi_sub[i,1]-candi_sub[j,3])**2)>25:
                    continue
                for t in range(128):
#                    sift_diff_tmp = (descriptor_master[i,t]-descriptor_slave[j,t])**2   # Euclidean distance
#                    sift_diff_sum += sift_diff_tmp
                        
                    sift_diff_tmp_1 = descriptor_master[i,t]*descriptor_slave[j,t]       # Cosine distance
                    sift_diff_sum_1 += sift_diff_tmp_1                    
                    sift_diff_tmp_2 = descriptor_master[i,t]**2
                    sift_diff_sum_2 += sift_diff_tmp_2                    
                    sift_diff_tmp_3 = descriptor_slave[j,t]**2
                    sift_diff_sum_3 += sift_diff_tmp_3
                    
                sift_diff[j,0] = sift_diff_sum_1 / (np.sqrt(sift_diff_sum_2) * np.sqrt(sift_diff_sum_3))                
#                sift_diff[j,0] = np.sqrt(sift_diff_sum)
                sift_diff[j,1] = i
                sift_diff[j,2] = j
            
            sift_diff = sift_diff[~(sift_diff[:,0:2]==0).all(1)]
            sift_diff = sift_diff[sift_diff[:,0].argsort()]
#            sift_diff_min = sift_diff[0,:].reshape((1,3))
            sift_diff_min = sift_diff[-1,:].reshape((1,3))
            cor_points = np.row_stack((cor_points, sift_diff_min[:,1:3]))
        
        if k == 1:
            index_ma = cor_points[:,1].flatten()
            index_sl = cor_points[:,0].flatten()
            for s in range (cor_points.shape[0]):
                cor_points_sift_ma[:,0:2] = candi_sub[int(index_ma[s]),0:2]
                cor_points_sift_sl[:,0:2] = candi_sub[int(index_sl[s]),2:4]
                cor_points_sift_tmp = np.column_stack((cor_points_sift_ma,cor_points_sift_sl))
                cor_points_sift_2 = np.row_stack((cor_points_sift_2,cor_points_sift_tmp))
        else:
            index_ma = cor_points[:,0].flatten()
            index_sl = cor_points[:,1].flatten()
            for s in range (cor_points.shape[0]):
                cor_points_sift_ma[:,0:2] = candi_sub[int(index_ma[s]),0:2]
                cor_points_sift_sl[:,0:2] = candi_sub[int(index_sl[s]),2:4]
                cor_points_sift_tmp = np.column_stack((cor_points_sift_ma,cor_points_sift_sl))
                cor_points_sift_1 = np.row_stack((cor_points_sift_1,cor_points_sift_tmp))
        
    # Select detected common corresponding pair
    for i in range(cor_points_sift_1.shape[0]):
        for j in range(cor_points_sift_2.shape[0]):
            if (cor_points_sift_1[i] == cor_points_sift_2[j]).all:
                cor_points_sift = np.row_stack((cor_points_sift,cor_points_sift_1[i]))
                break      
    
    cor_points_sift = np.unique(cor_points_sift, axis=0)                       # Remove repeated detection points    
        
    return cor_points_sift    

def blunder_detection(cor_points):
    cor_points_in = np.zeros((0,4))
    for i in range(cor_points.shape[0]):
        if (np.abs(cor_points[i,0]-cor_points[i,2])<1
            and np.abs(cor_points[i,1]-cor_points[i,3])<1):
            cor_points_tmp = cor_points[i,:]
            cor_points_in = np.row_stack((cor_points_in,cor_points_tmp))
            
    return cor_points_in
                
def get_flow_vec(cor_points):
    flow_vec_x = (cor_points[:,0]-cor_points[:,2]).reshape(cor_points.shape[0],1)
    flow_vec_y = (cor_points[:,1]-cor_points[:,3]).reshape(cor_points.shape[0],1)
    flow_vec = np.column_stack((cor_points[:,0:2] ,flow_vec_x, flow_vec_y))
    
    fig, ax = plt.subplots()
    plt.hist(np.ravel(flow_vec[:,2:4]), bins=50, range=(np.min(flow_vec[:,2:4]),np.max(flow_vec[:,2:4])))
    plt.show()
    
    return flow_vec

def visualization(cor_points):
    ma_cor = cor_points[:,0:2]
    sl_cor = cor_points[:,2:4]
    index_2 = np.arange(0,cor_points.shape[0],1).T
    
    fig, ax = plt.subplots()
    plt.gray()   
    plot_matches(ax, my_img_master, my_img_slave, ma_cor, sl_cor,
             np.column_stack((index_2, index_2)), matches_color='r', alignment='vertical')
    ax.axis('off')
    plt.show()
    
def visualization_vec(image, flow_vec, name):
    
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=cm.gray)
    plt.axis('tight')
    q = ax.quiver(flow_vec[:,1], flow_vec[:, 0], flow_vec[:, 2], -flow_vec[:, 3],
                  color='red',
                  width=0.0015, headwidth=3)
    plt.savefig("./images/flow_vec_quiver/" + name)
    plt.show() 
    
if __name__== '__main__':
    my_img_master = io.imread('./kerman_data/master.tiff')
    my_img_slave = io.imread('./kerman_data/slave.tiff')

#    my_img_master = io.imread('./planet/master.tiff').astype(np.float32)
#    my_img_slave = io.imread('./planet/slave.tiff').astype(np.float32)
#    
#    my_img_master = 255*(my_img_master-np.min(my_img_master))/(np.max(my_img_master) - np.min(my_img_master))#500)#
##    my_img_slave = 255*(my_img_slave-np.min(my_img_slave))/(np.max(my_img_slave) -np.min(my_img_slave)) # 500)# 
##    my_img_master = 255*(my_img_master-920)/(1720 - np.min(my_img_master))
#    my_img_slave = 255*(my_img_slave-1000)/(1720 - 1000)
    
    cornerness_matrix_ma, coords_subpix_ma = point_detectors_B.tiled_point_detection(
            my_img_master, partition=8, method = "foerstner_skimage", 
            min_distance=5, num_peaks=200)
    cornerness_matrix_sl, coords_subpix_sl = point_detectors_B.tiled_point_detection(
            my_img_slave, partition=8, method = "foerstner_skimage", 
            min_distance=5, num_peaks=200)
    
    # Extract grey value
    matching_matrix_ma = extract_greyval(cornerness_matrix_ma, my_img_master)  
    matching_matrix_sl = extract_greyval(cornerness_matrix_sl, my_img_slave)
    
    # Find corresponding point candidates
    candidate, candi_sub = point_based_matching(matching_matrix_ma, matching_matrix_sl,
                                                coords_subpix_ma, coords_subpix_sl)
    
    # Apply SIFT matching based on sub-pixel
    descriptor_master, ori_master = sift.sift_descriptor(my_img_master, candi_sub[:,0:2])
    descriptor_slave, ori_slave = sift.sift_descriptor(my_img_slave, candi_sub[:,2:4])    
    cor_points = sift_matching(descriptor_master, descriptor_slave, candi_sub)
    
    # Blunder detection
    cor_points_in = blunder_detection(cor_points)
    
    # Compute the whole flow vector
    flow_vec = get_flow_vec(cor_points)
    np.savetxt('Flow vector (all)', flow_vec)
    
#    visualization_vec(my_img_master, flow_vec, 'flow_vec')
   
    # Compute the flow vector after blunder detection
    flow_vec_in = get_flow_vec(cor_points_in)
    np.savetxt('Flow vector (inlier)', flow_vec_in)
    
#    visualization_vec(my_img_master, flow_vec_in, 'flow_vec_in')
    
    fig, ax = plt.subplots()
    #ax.set(title=p.tostring)     
    ax.imshow(my_img_master, interpolation='nearest', cmap=cm.gray)
    ax.plot(flow_vec_in[:, 1], flow_vec_in[:, 0], 'ro', markersize=4)
    plt.show()

