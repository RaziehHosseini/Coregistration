# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:52:47 2019

@author: razieh
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import IPython.display
from skimage.feature import plot_matches
from matplotlib import cm

import read_data


def display(mstr_amp, slv_amp, lineno, loc):
    PATH = "./images/original/"
    ### Master
    fig = plt.figure('master')
    ax = fig.add_subplot(111)
    cax = ax.imshow(mstr_amp, interpolation='nearest', cmap=cm.gray)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, mstr_amp.max()])
    plt.axis('tight')
    plt.savefig(PATH +"(master_" + loc + ")"+ lineno)
    ### Slave
    fig = plt.figure('slave')
    ax = fig.add_subplot(111)
    cax = ax.imshow(slv_amp, interpolation='nearest', cmap=cm.gray)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, slv_amp.max()])
    plt.axis('tight')
    plt.savefig(PATH +"(salve_" + loc + ")" + lineno)
    plt.show()


def draw_points(img, p, lineno, loc, name, method_name, sp=None, counter = "0"): # p:detected points sp:detected points (sub-pixels)
    PATH = "./images/points/"
    fig, ax = plt.subplots()
    #ax.set(title=p.tostring)     
    ax.imshow(img, interpolation='nearest', cmap=cm.gray)
    ax.plot(p[:, 1], p[:, 0], 'ro', markersize=4)
    if sp.any()!= None:
        ax.plot(sp[:, 1], sp[:, 0], '.b', markersize=2)
    plt.savefig (PATH+name+loc+'_'+method_name+ lineno + np.str(counter))
    
    plt.show()
    
def draw_fv(img, coord, direction, PATH):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=cm.gray)
    plt.axis('tight')
    plt.title(PATH)
    q = ax.quiver(coord[:,0], coord[:, 1], direction[:, 0], direction[:, 1],
                  color='red',
                  width=0.0012, headwidth=3,
                  scale=None)
    PATH = "./images/flow_vec_quiver/" + PATH
    plt.savefig (PATH)
    plt.show()
def plot_point_matches(src, dest, img1, img2):
    index = np.arange(0,src.shape[0],1).T
    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()
    plot_matches(ax, img1, img2, src, dest, 
                 np.column_stack((index, index)), matches_color='b', alignment='vertical')
    ax.axis('off')
    ax.set_title('Correct correspondences')

def plot_hist(vec, title='No title', labels=None):
    plt.figure(title)
    if labels != None:
        plt.hist(vec, bins=100, label=labels)
        plt.legend()
    else :
        plt.hist(vec, bins=100)
        
    plt.xlabel('flow_vec_magnitude')
    plt.ylabel('freq')
    if title!=None:
        plt.title(title)
    plt.savefig('./histPlots/'+title)
    plt.show()


# following function is taken from Image Analysis I 
# Institute of photogrametry and Informatics - leibniz university of hannover
def imshow3D(*I):
    """Shows the array representation of one or more images in a jupyter notebook.

        Parameters
        ----------
        I : ndarray of float64
            Array representation of an image
            Concatenates multiple images

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """

    if len(I) == 1:
        I = I[0]
    else:
        channels = [i.shape[2] for i in I]
        heights = [i.shape[0] for i in I]
        max_height = max(heights)
        max_channels = max(channels)

        if min(channels) != max_channels:  # if one image has three channels ..
            I = list(I)
            for i in range(len(I)):
                dim = channels[i]
                if dim == 1:  # .. and another has one channel ..
                    I[i] = np.dstack((I[i], I[i], I[i]))  # .. expand that image to three channels!

        if min(heights) != max_height:  # if heights of some images differ ..
            I = list(I)
            for i in range(len(I)):
                h, w, d = I[i].shape
                if h < max_height:  # .. expand by 'white' rows!
                    I_expanded = np.ones((max_height, w, d), dtype=np.float64) * 255
                    I_expanded[:h, :, :] = I[i]
                    I[i] = I_expanded

        seperator = np.ones((max_height, 3, max_channels), dtype=np.float64) * 255
        seperator[:, 1, :] *= 0
        I_sep = []
        for i in range(len(I)):
            I_sep.append(I[i])
            if i < (len(I) - 1):
                I_sep.append(seperator)
        I = np.hstack(I_sep)  # stack all images horizontally

    assert I.ndim == 3
    h, w, d = I.shape
    assert d in {1, 3}
    if d == 1:
        I = I.reshape(h, w)
    IPython.display.display(Image.fromarray(I.astype(np.ubyte)))


#
##img1 , img2 = ri.read_img('Master.mat','Master','Slave.mat','Slave')
#img1 , img2 = read_data.read_img_nocrop('master_crop.mat','master','slave_crop.mat','slave', './kerman_data/')
#
#img = np.array(img1)
#gray = Image.fromarray(img1)
#
#h, w = img.shape
#img1 = img1.reshape((h, w, 1)).astype(np.float64)
#imshow3D(img1)