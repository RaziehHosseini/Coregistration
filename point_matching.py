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
    
def point_based_matching(matrix_ma, matrix_sl, coords_subpix_ma, coords_subpix_sl, weight=0.5):
    # Initialize the variable
    num_ma = matrix_ma.shape[0]
    num_sl = matrix_sl.shape[0]
    cor_points_1 = np.zeros((0,6))      # Corresponding points from ma --> sl
    cor_points_2 = np.zeros((0,6))      # Corresponding points from sl --> ma
    cor_points = np.zeros((0,6))        # Final corresponding points
    loss = np.zeros((max(num_ma,num_sl),7))
    loss_min = np.zeros((1,7))
    
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
                lo = 1/(np.exp(np.sqrt(x_lo_diff**2 + y_lo_diff**2)))
#                lo = 1/(np.sqrt(x_lo_diff**2 + y_lo_diff**2)+0.001)                
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

            if (loss[:,0:5]==0).all():                                         # If no score, return 0
                loss = np.zeros((1,7))
            else:
                loss = loss[~(loss[:,0:5]==0).all(1)]                          # Remove all 0 value
                loss = loss[loss[:,0].argsort()]                               # Resort the score value
                loss_min = loss[0,:].reshape((1,7))                            # Best score for matching
                
            if k == 1:                                                         # Cor-points from sl-->ma
                cor_tmp = loss_min[0,1:7].reshape((1,6))
                cor_points_2 = np.row_stack((cor_points_2,cor_tmp))
                cor_points_2 = cor_points_2[~(cor_points_2[:,0:4]==0).all(1)]         # Remove all 0 rows
            else:                                                              # Cor-points from ma-->sl
                cor_tmp = loss_min[0,1:7].reshape((1,6))
                cor_points_1 = np.row_stack((cor_points_1,cor_tmp))
                cor_points_1 = cor_points_1[~(cor_points_1[:,0:4]==0).all(1)]
            loss = np.zeros((max(num_ma,num_sl),7))                        # Initialize the score
    
    # Select detected common corresponding pair
    for i in range(cor_points_1.shape[0]):
        for j in range(cor_points_2.shape[0]):
            if (cor_points_1[i] == cor_points_2[j]).all:
                cor_points = np.row_stack((cor_points,cor_points_1[i]))
                break
            
    cor_points = np.unique(cor_points, axis=0)                                 # Remove repeated detection points
    
    print("\n There are: ",cor_points.shape[0], "pairs of corresponding points") # Print the results
    
    # Get subpixel accuracy
    cor_sub_tmp = np.zeros((1,4))
    cor_sub = np.zeros((0,4))
    for i in range(cor_points.shape[0]):
        cor_sub_tmp[:,0:2] = coords_subpix_ma[int(cor_points[i,4]),:]
        cor_sub_tmp[:,2:4] = coords_subpix_sl[int(cor_points[i,5]),:]
        cor_sub = np.row_stack((cor_sub, cor_sub_tmp))
    
    cor_sub = cor_sub[~np.isnan(cor_sub).any(1)]
    
    # Blunder detection
    cor_points_in = np.zeros((0,4))    
    for i in range(cor_sub.shape[0]):
        if (np.abs(cor_sub[i,0]-cor_sub[i,2])<1
            and np.abs(cor_sub[i,1]-cor_sub[i,3])<1):
            cor_points_in_tmp = cor_sub[i,0:4]
            cor_points_in = np.row_stack((cor_points_in, cor_points_in_tmp))
    
    print(" There are: ",cor_points_in.shape[0], "pairs of inlier corresponding points")
        
    return cor_points, cor_sub, cor_points_in
                
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
    q = ax.quiver(flow_vec[:,1], flow_vec[:, 0], flow_vec[:, 3], flow_vec[:, 2],
                  color='red',
                  width=0.0018, headwidth=3,
                  scale=None)
    plt.savefig("./images/flow_vec_quiver/" + name)
    plt.show() 
    
if __name__== '__main__':
    my_img_master = io.imread('./kerman_data/master.tiff')
    my_img_slave = io.imread('./kerman_data/slave.tiff')
    
    cornerness_matrix_ma, coords_subpix_ma = point_detectors_B.tiled_point_detection(
            my_img_master, partition=8, method = "foerstner_skimage", 
            min_distance=5, num_peaks=500)
    cornerness_matrix_sl, coords_subpix_sl = point_detectors_B.tiled_point_detection(
            my_img_slave, partition=8, method = "foerstner_skimage", 
            min_distance=5, num_peaks=500)
    
    # Extract grey value
    matching_matrix_ma = extract_greyval(cornerness_matrix_ma, my_img_master)  
    matching_matrix_sl = extract_greyval(cornerness_matrix_sl, my_img_slave)
    
    # Find the corresponding points and detect inliers
    cor_points, cor_sub, cor_points_in= point_based_matching(matching_matrix_ma, matching_matrix_sl,
                                                    coords_subpix_ma, coords_subpix_sl)
    
    # Compute the whole flow vector
    flow_vec = get_flow_vec(cor_points)
    np.savetxt('Flow vector (all)', flow_vec)
    
#    visualization_vec(my_img_master, flow_vec, 'flow_vec')
    
    # Compute the flow vector in subpixel wise
    flow_vec_sub = get_flow_vec(cor_sub)
#    visualization(cor_sub)
#    np.savetxt('Flow vector (inlier)', cor_sub)
#    visualization_vec(my_img_master, flow_vec_in)

    # Compute the flow vector after blunder detection
    flow_vec_in = get_flow_vec(cor_points_in)
    np.savetxt('Flow vector (inlier)', flow_vec_in)
    
#    visualization_vec(my_img_master, flow_vec_in, 'flow_vec_in')

    fig, ax = plt.subplots()
    #ax.set(title=p.tostring)     
    ax.imshow(my_img_master, interpolation='nearest', cmap=cm.gray)
    ax.plot(flow_vec_in[:, 1], flow_vec_in[:, 0], 'ro', markersize=4)
    plt.show()
    
    