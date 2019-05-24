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

#def area_based_ransac():
    

class Matcher:
    def __init__(self, master, slave, master_points, slave_points, g_sigma=1 ,neighbour_weights=None, similarity= 'SSD'):
        
        self.similarity = similarity
        self.master = master
        self.slave = slave
        self.s_points = slave_points
        self.m_points = master_points
        self.neighbour_weights = neighbour_weights
        self.sigma = g_sigma


    """==================     Similarity measure   =======================
       ============================================================"""
    #@property
    def SSD(self, master_win, slave_win):
        ssd = np.sum(self.neighbour_weights * (master_win-slave_win)**2)
        return ssd
    
    
#    def cross_correlation(self, master_win, slave_win):
#        return cross_correlation
        
    
    
    """==================      Matching strategies   =======================
       ====================================================================="""
       
    dict_func = {"SSD": SSD} #, "cross_correlation":cross_correlation}
       
    def match_window_based(self,master_point, window_size=3):
        
        best_match = [] # To store the similarities (SSDs, or other measures)
        assert ((window_size+1)%2==0), "Even window size is not possible!"
        
        offset = np.int((window_size-1)/2)
        r,c = np.round(master_point).astype(np.intp)
        master_win = self.master[r-offset:r+offset+1,c-offset:c+offset+1]
        self.neighbour_weights = gaussian_weights(window_size, self.sigma)
        
        for x, y in self.s_points:
            slave_win = self.slave[x-offset:x+offset+1,y-offset:y+offset+1]                     # in case of more dimention use: y-window_size:y+window_size+1, :]) 
            similarity_method = self.dict_func.get(self.similarity)
            best_match.append(similarity_method(self, master_win, slave_win))
            
        
        min_idx = np.argmin(best_match)    
        return self.s_points[min_idx]
    
#    def match_area_based(self, window_size):                                  # TODO
#        """area_based means that we define the searching area
#        in the second image (slave) we don't compare all the points
#        """
    
        
        
    #def match_point_wise():                                                   # TODO
        
        
    
    def match(self):
        src = []
        dst = []
        # extend the dimension of master and slave or 
        # only extend for points, for later one matching only could be done then
        # point wise or extention must be done for neighbouring pixels also
                                                                                # TODO (later we have to )
        for master_point in self.m_points:
            src.append(master_point)
            dst.append(self.match_window_based(master_point, window_size=3))
        
        return src, dst


#if __name__== '__main__':
#
#    my_img_master = io.imread('./noiseless_data/master.tiff')
#    my_img_slave = io.imread('./noiseless_data/slave.tiff')
#    
#    matching_matrix_ma = point_detectors.tiled_point_detection(
#            my_img_master, partition=2, method = "foerstner_skimage")
#    matching_matrix_sl = point_detectors.tiled_point_detection(
#            my_img_slave, partition=2, method = "foerstner_skimage")
#    
#    cor_points = point_based_matching(matching_matrix_ma, matching_matrix_sl, 10, 10, 0.2)

#  