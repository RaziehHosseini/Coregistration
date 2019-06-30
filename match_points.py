# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:42:45 2019

@author: p-sem-2019
"""

# imports

import numpy as np 
import point_detectors
from skimage import io
"""==================     helper   =======================
   ============================================================"""  

def gaussian_weights(window_size, sigma=1):

    #assert np.int(sigma) > 0.0 , "Std. deviation must be positive!"
    assert ((window_size+1)%2==0), "Even window size is not possible!"
        
    offset = np.int((window_size-1)/2)
    y, x = np.mgrid[-offset:offset+1, -offset:offset+1]
    g = np.zeros((window_size,window_size), dtype=np.float64)
    twoSigmaSqu = (sigma**2*2)
    g[:] = 1.0 /(np.pi*twoSigmaSqu)*np.exp(-(x**2+y**2)/float(twoSigmaSqu))
    return g

def extract_greyval(cornerness_matrix, image):
    points_num = int(cornerness_matrix.shape[0])
    coord_x = cornerness_matrix[:,0].reshape((points_num,1)).astype('int64')
    coord_y = cornerness_matrix[:,1].reshape((points_num,1)).astype('int64')
    gray_val = image[coord_x,coord_y].reshape((points_num,1))
    matching_matrix = np.hstack((coord_x, coord_y, gray_val))
    
    return matching_matrix
	
#def area_based_ransac():                                                      # TODO
    

class Matcher:
    def __init__(self, master, slave, master_points, 
                 slave_points, scorer_beta, g_sigma=1 ,neighbour_weights=None, similarity= 'SSD'):
        
        self.similarity = similarity
        self.master = master
        self.slave = slave
        self.s_points = slave_points
        self.m_points = master_points
        self.neighbour_weights = neighbour_weights
        self.sigma = g_sigma
        self.scorer_beta = scorer_beta



    """==================     Similarity measure   =======================
       ============================================================"""
    
    """============= source:
        https://numerics.mathdotnet.com/Distance.html====="""
    def SAD (self, v1, v2):
        """ sum of absolute difference 
        *** [Manhattan distance] or [L1 norm] or [minkowski with p=1]"""
        return np.sum(np.abs(v2-v1))
    
    def SSD(self, master_win, slave_win): # win or vector doesn't make any diff
        """sum of squared difference"""
        return np.sum( (master_win-slave_win)**2)
	
    def MAE (self, v1, v2):
        """ mean absolute error [normalized SAD]"""
        n = v1.shape[0]
        return (np.sum(np.abs(v2-v1)))/n
    
    def MSE (self, v1, v2):
        """ mean squared error [normalized version of SSD]"""
        n = v1.shape[0]
        return (np.sum(np.abs(v2-v1)))/n
    
    def euclidean_dist(self, sift_v1, sift_v2):
        """ squared sum of squared differences np.sqrt(SSD)"""
        return np.sqrt(np.sum((sift_v1-sift_v2)**2))
    
    def chebyshev_dist(self, u, v):
        """ L_inf norm [minkowski distance where p  = inf]"""
        return max (abs(u-v))
    
    def minkowski_dist(self, u, v, p):
        """ L_p norm """
        return (np.sum((np.abs(u-v))**p))**(1/p)
    
    def canberra_distance(self, u, v):
        """ weighted version of Manhattan distance, often used for data 
        scattered around an origin. very sensitive for values close to zero"""
        return np.sum((np.abs(u-v))/(np.abs(u)+np.abs(v)))
    
    def cosine_dist (self, u, v):
        """dot product scaled by product of l2 norm of vectors
        it represents the angular distance of two vectors ignoring their scale [amplitude]"""
        return np.sum(u*v)/(np.sqrt(np.sum(u**2))* np.sqrt(np.sum(v**2)))
        
    """=========================   NO source        ========================"""
    def WSSD(self, master_win, slave_win):
        """ sum of squared differences with gaussian weight"""
        wssd = np.sum( self.neighbour_weights *((master_win-slave_win)**2))
        return wssd
    
    def cross_correlation_dist(self, v, u):
        """ As it is a distance 1-corr is considered
        u and v can be window around the point or can be 2 SIFT vectors"""
        u_mean = np.average(u)
        v_mean = np.average(v)
        ut = u-u_mean
        vt = v-v_mean
        return 1-np.sum(((ut*vt)/(np.linalg.norm(ut)*np.linalg.norm(vt))))
      
    def distance_score_wrt_position(self, r, c, x, y, distance, beta):
        dx = (r-x)**2
        dy = (c-y)**2
        z = 1/(np.exp(np.sqrt(dx+dy)))
        s = beta*(z) + (1-beta)*distance
        return s    
        
    """==================      Matching strategies   =======================
       ====================================================================="""
    dict_func = {"SSD": SSD,
                 "SAD": SAD,
                 "MAE":MAE,
                 "MSE":MSE,
                 "euclidean_dist": euclidean_dist,
                 "chebyshev_dist":chebyshev_dist,
                 "minkowski_dist":minkowski_dist,
                 "canberra_distance": canberra_distance,
                 "cosine_dist":cosine_dist,
                 "WSSD":WSSD,
                 "cross_correlation_dist": cross_correlation_dist}
         
    def match_window_based(self, master_point, window_size, **kwargs):
        
        distance_score = [] # To store the similarities (SSDs, or other measures)
        assert ((window_size+1)%2==0), "Even window size is not possible!"
        
        offset = np.int((window_size-1)/2)
        r,c = np.round(master_point).astype(np.intp)
        if r <= offset or c <= offset or r >= self.slave.shape[0]-offset or c >= self.slave.shape[1]-offset:
            distance_score.append(255**2)
        else:
            master_win = self.master[r-offset:r+offset+1,c-offset:c+offset+1]
            self.neighbour_weights = gaussian_weights(window_size, self.sigma)
			
            for x, y in self.s_points:
                if x <= offset or y <= offset or x >= self.slave.shape[0]-offset or y >= self.slave.shape[1]-offset:
                    distance_score.append(255**2)
					
                else:
                    #print(' --- ')
                    slave_win = self.slave[x-offset:x+offset+1,y-offset:y+offset+1]                     
                    similarity_method = self.dict_func.get(self.similarity)
                    feature_distance = similarity_method(self, master_win, slave_win)
                    #print(feature_distance)
                    #score = self.distance_score_wrt_position(r, c, x, y,feature_distance, self.scorer_beta)
                    #print(score)
                    # if np.sqrt((r-x)**2+(c-y)**2) < 50:
                    #     print(feature_distance)
                    distance_score.append(feature_distance)
        #print('min',min(distance_score))
        min_idx = np.argmin(distance_score)    # smaller distance =>  smaller similarity value=> better correspondence
        return self.s_points[min_idx]
    
    def match_area_based(self, area_size):                                     # TODO
        """area_based means that we define the searching area
        in the second image (slave) we don't compare all the points
        """
        
    def match_point_wise_nD(self, master_point, index_mp, descriptor_master, descriptor_slave, **kwargs):
        """nD: multi dimensional matcher (SIFT): 
           not only distance of grey_value but for feature vectors like SIFT"""
           # check direction
        if self.m_points.shape[0] != descriptor_master.shape[0]:
             temp = descriptor_master
             descriptor_master = descriptor_slave
             descriptor_slave = temp
           
        distance_score = []
        for index, (x, y) in enumerate(self.s_points):
            similarity_method = self.dict_func.get(self.similarity)
            master_sift_v = descriptor_master[index_mp,:]
            slave_sift_v = descriptor_slave[index,:]
            feature_distance = similarity_method(self, master_sift_v, slave_sift_v)
            distance_score.append(feature_distance)
        min_idx = np.argmin(distance_score)
        return self.s_points[min_idx]
    
    def cross_correlation_matcher(self, master_point, window_size, **kwargs):
        """ correlation_coefficients in whole image"""
        assert ((window_size+1)%2==0), "Even window size is not possible!"
        offset = np.int((window_size-1)/2)
        distance_score = np.zeros((0,3), dtype=np.float16)
        r,c = np.round(master_point).astype(np.int)
        if r <= offset or c <= offset or r >= self.slave.shape[0]-offset or c >= self.slave.shape[1]-offset:
            distance_score[:,r,c] = 255**2
        else:
            master_win = self.master[r-offset:r+offset+1,c-offset:c+offset+1]			
            for (x, y) in zip(range(self.slave.shape[0]), range(self.slave.shape[1])):
                if x <= offset or y <= offset or x >= self.slave.shape[0]-offset or y >= self.slave.shape[1]-offset:
                    distance_score[:,r,c] = 255**2
					
                else:
                    slave_win = self.slave[x-offset:x+offset+1,y-offset:y+offset+1]                     
                    feature_distance = self.cross_correlation_dist(master_win, slave_win)
                    print('-------------------')
                    print(feature_distance)
                    distance_score = np.row_stack(distance_score,[feature_distance, x, y])
        min_idx = np.argmin(distance_score[0,:,:])    # smaller distance =>  smaller similarity value=> better correspondence
        return distance_score[min_idx]
        
        
    def match(self, method, **kwargs):
        src = []
        dst = []
    
        if method == "match_window_based":
            for master_point in self.m_points:
                src.append(master_point)
                dst.append(self.match_window_based(master_point, **kwargs))
        
        elif method == "match_point_wise_nD":
            for index_mp,(master_pointx, master_pointy) in enumerate(self.m_points[:]):
                master_point = [master_pointx, master_pointy]
                src.append(master_point)
                dst.append(self.match_point_wise_nD(master_point, index_mp, **kwargs))
        
        elif method == "match_area_based":
            for index_mp,(master_pointx, master_pointy) in enumerate(self.m_points[:]):
                master_point = [master_pointx, master_pointy]
                src.append(master_point)
                dst.append(self.match_area_based(master_point, index_mp, **kwargs))
        
        elif method == "cross_correlation_matcher":
            for master_point in self.m_points:
                src.append(master_point)
                matched_point = self.cross_correlation_matcher(master_point, **kwargs)
                dst.append(master_point)
        
        return np.array(src), np.array(dst)

#  