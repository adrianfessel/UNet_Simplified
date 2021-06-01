# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:18:15 2021

@author: Adrian
"""

import os
import cv2

import numpy as np

import scipy.signal as signal

from tqdm import tqdm
from skimage.filters import threshold_otsu


def get_growth_direction(path, first=None, last=None, increment=None, dt=48, binning=False):
    """
    Function to determine the growth direction of a binary object (slime mold)
    in a sequence of images. The direction is determined by computing the
    difference between binary frames that are a certain interval apart.

    Parameters
    ----------
    path : string
        directory containing image sequence
    first : int, optional
        first frame; the default is None
    last : int, optional
        last frame; the default is None
    increment : int, optional
        increment; the default is None
    dt : int, optional
        interval for computing the difference of binary images; the default is 48.
    binning : int, optional
        binning factor; the default is False

    Returns
    -------
    phi_p : numpy array
        growth direction, measured with reference to the past centroid
    phi_n : numpy array
        retraction direction, measured with reference to the past centroid

    """
    
    path_Probability = path + '_Probability'
    
    frames_Probability = [Frame for Frame in os.listdir(path_Probability) if '.tif' in Frame or '.png' in Frame or '.jpeg' in Frame or '.jpg' in Frame]
    frames_Probability = frames_Probability[first:last:increment]
    
    phi_p, phi_n = [], []    
    B_past = []
    XC_past, YC_past = [], []
    
    for t, frame_P in tqdm(enumerate(frames_Probability)):
        
        P = cv2.imread(os.path.join(path_Probability,frame_P),cv2.IMREAD_COLOR)
        
        if binning:
            dsize = (int(P.shape[1]/binning),int(P.shape[0]/binning))
            P = cv2.resize(P, dsize, cv2.INTER_CUBIC)
        
        info = np.iinfo(P.dtype)
        P = np.float32(P)/info.max    
        B = P[:,:,1] < threshold_otsu(P[:,:,1])
        
        YB, XB = B.nonzero()
        # XC, YC = np.mean(XB), np.mean(YB)
        
        XC_past.append(np.mean(XB))
        YC_past.append(np.mean(YB))
        
        B_past.append(B)
    
        if t > dt:
            
            B_diff = np.asarray(B, dtype=np.float32) - np.asarray(B_past.pop(0), dtype=np.float32)

            pos = B_diff > 0
            neg = B_diff < 0

            Yp, Xp = pos.nonzero()
            Yn, Xn = neg.nonzero()
            
            XCn, YCn = np.mean(Xn), np.mean(Yn)
            XCp, YCp = np.mean(Xp), np.mean(Yp)
            
            XC, YC = XC_past.pop(0), YC_past.pop(0)
            
            dXn = XCn - XC
            dYn = YCn - YC      
            
            dXp = XCp - XC
            dYp = YCp - YC  
            
            phi_p.append(np.arctan2(dYp, -dXp))
            phi_n.append(np.arctan2(dYn, -dXn))
            
    phi_p = np.asarray(phi_p)
    phi_n = np.asarray(phi_n)
       
    phi_p[phi_p < 0] = phi_p[phi_p < 0] + np.pi
    phi_p[phi_p > np.pi] = phi_p[phi_p > np.pi] - np.pi
    
    phi_n[phi_n < 0] = phi_n[phi_n < 0] + np.pi
    phi_n[phi_n > np.pi] = phi_n[phi_n > np.pi] - np.pi
    
    return phi_p, phi_n


def get_AIC(path, first=None, last=None, increment=None, binning=False, centroid=False):
    """
    Function to compute area, intensity and centroid of regions belonging to
    the discrete levels in a three-level presegmented image. Intended for
    processing a directory of images.

    Parameters
    ----------
    path : string
        directory containing image sequence
    first : int, optional
        first frame; the default is None
    last : int, optional
        last frame; the default is None
    increment : int, optional
        increment; the default is None
    binning : int, optional
        binning factor; the default is False
    centroid : bool, optional
        controls whether the region centroid is computed; the default is False

    Returns
    -------
    data : dict
        dict of results, keyed by type:
            'AM', 'AN', 'AF' : areas of mold, network, front
            'IM', 'IN', 'IF' : intensities of mold, network, front
            'XM', 'XN', 'XF' : centroid x-coordinate of mold, network, front
            'YM', 'YN', 'YF' : centroid y-coordinate of mold, network, front

    """
    
    path_Probability = path + '_Probability'
    path_Gray = path
    
    frames_Probability = [frame for frame in os.listdir(path_Probability) if '.tif' in frame or '.png' in frame or '.jpeg' in frame or '.jpg' in frame]
    frames_Gray = [frame for frame in os.listdir(path_Gray) if '.tif' in frame or '.png' in frame or '.jpeg' in frame or '.jpg' in frame]

    ext_G, ext_P = frames_Gray[0].split('.')[1], frames_Probability[0].split('.')[1]

    frames_Gray = [frame.split('.')[0] for frame in frames_Gray]
    frames_Probability = [frame.split('.')[0] for frame in frames_Probability]

    frames_Gray = [frame for frame in frames_Gray if frame in frames_Probability]

    frames_Gray = [frame+'.'+ext_G for frame in frames_Gray]
    frames_Probability = [frame+'.'+ext_P for frame in frames_Probability]

    frames_Gray = frames_Gray[first:last:increment]
    frames_Probability = frames_Probability[first:last:increment]

    data = {'AM':np.zeros(len(frames_Gray)), 'AN':np.zeros(len(frames_Gray)), 'AF':np.zeros(len(frames_Gray)), 
            'IM':np.zeros(len(frames_Gray)), 'IN':np.zeros(len(frames_Gray)), 'IF':np.zeros(len(frames_Gray))}
    
    if centroid:
    
        data['XM'] = np.zeros(len(frames_Gray))
        data['XN'] = np.zeros(len(frames_Gray))
        data['XF'] = np.zeros(len(frames_Gray))
        data['YM'] = np.zeros(len(frames_Gray))
        data['YN'] = np.zeros(len(frames_Gray))
        data['YF'] = np.zeros(len(frames_Gray))

    for i, (frame_G, frame_P) in tqdm(enumerate(zip(frames_Gray, frames_Probability)), total=len(frames_Gray)):
        
        I = cv2.imread(os.path.join(path_Gray,frame_G),cv2.IMREAD_GRAYSCALE)
        P = cv2.imread(os.path.join(path_Probability,frame_P),cv2.IMREAD_COLOR)
        
        if binning or I.shape != P.shape:
            if not binning:
                binning = 1
                    
            dsize = (int(np.min([I.shape[1], P.shape[1]])/binning), int(np.min([I.shape[0], P.shape[0]])/binning))
            
            I = cv2.resize(I, dsize, cv2.INTER_CUBIC)
            P = cv2.resize(P, dsize, cv2.INTER_CUBIC)
        
        info = np.iinfo(P.dtype)
        P = np.float32(P)/info.max
        
        T = threshold_otsu(P[:,:,2])
        T = T if T > 0.1 else 0.25
        
        B = np.bitwise_and(P[:,:,1] > P[:,:,0], P[:,:,1] > P[:,:,2])
        F = np.bitwise_and(~B, P[:,:,2] < T)
        N = np.bitwise_and(~B, ~F)
        
        for X, S in zip([N,F,~B],['N','F','M']):

            data['A'+S][i] = np.sum(X)
            data['I'+S][i] = np.asarray(np.ma.array(I, mask = ~X).mean(axis=(0,1)))
            
            if centroid:
                coords = np.where(X)
                data['X'+S][i] = np.mean(coords[1])
                data['Y'+S][i] = np.mean(coords[0])
    
    return data


def crossings(y):
    """
    Function to determine the zero crossings and crossing type 
    (positive/negative) of a function. 

    Parameters
    ----------
    y : numpy array
        input function

    Returns
    -------
    p : numpy array
        indices of positive zero crossings
    n : numpy array
        indices of negative zero crossings

    """
    
    s = np.sign(y)
    
    p = np.argwhere(np.diff(s) == 2)
    n = np.argwhere(np.diff(s) == -2)

    return p, n


def fix_gaps(p, n, g):
    """
    Function to merge positive/negtaive crossings that are closer than a given
    threshold distance.

    Parameters
    ----------
    p : numpy array
        indices of positive zero crossings
    n : numpy array
        indices of negative zero crossings
    g : float
        threshold distance

    Returns
    -------
    p : numpy array
        indices of positive zero crossings
    n : numpy array
        indices of negative zero crossings

    """

    if len(p) > len(n):
        d = abs(p[:-1] - n)
        
        n = n[d>=g]
        p = p[np.append(d>=g, True)]
        
    elif len(n) > len(p):
        d = abs(n[:-1] - p)
        
        p = p[d>=g]
        n = n[np.append(d>=g, True)]
        
    elif len(n) == len(p):
        d = abs(p-n)
        
        p = p[d>=g]
        n = n[d>=g]
        
    return p, n


def det_frequency(x, y, g):
    """
    Determine the frequency of a function y(x) using the zero crossings
    method. Function must be centred around zero. Crossings closer than
    the threshold distance will be merged

    Parameters
    ----------
    x : numpy array
        function x coordinates
    y : numpy array
        function y coordinates
    g : float
        threshold distcne

    Returns
    -------
    f : float
        computed frequency

    """
    
    p, n = crossings(y)
    p, n = fix_gaps(p, n, g)
    
    return ((x[-1]-x[0])/len(n))/2+((x[-1]-x[0])/len(p))/2


def correlate(x1, x2, mode='coeff'):
    """
    Function to compute the cross correlation or correlation coefficient
    of two data series.

    Parameters
    ----------
    x1 : numpy array
        first series
    x2 : numpy array
        second series
    mode : string, optional
        operation mode; can be 'full' for computing the full cross correlation
        or 'coeff' for the coefficient; the default is 'coeff'

    Returns
    -------
    corr : numpy array
        cross correlation or coefficient

    """
    
    x1, x2 = x1 - x1.mean(), x2 - x2.mean()
    x1, x2 = x1 / x1.std(), x2 / x2.std()
    
    if mode == 'coeff':
        corr = signal.correlate(x1, x2, mode='valid')
    elif mode == 'full':
        corr = signal.correlate(x1, x2, mode='full')

    return corr/len(x1)


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    Path = 'E:/Movies_arbeit/2019_07_05_crop_3'
    
    frametime = 12
    first = None
    last = None
    increment = None
    binning = False
    pt = 0.1
    
    frametime = frametime*increment if increment else frametime
    gaplength = int(180/frametime)
    
    ### growth direction
    phi_p, phi_n = get_growth_direction(Path, first=2048, last=8192, increment=increment, binning=binning, dt=48)
    
    ### region size / gray values
    data = get_AIC(Path, first=first, last=last, increment=increment, binning=binning)
    
    # apply lambert-beer law 
    Tmax = np.nanmax([np.nanmax(data['IM']), np.nanmax(data['IN']), np.nanmax(data['IF'])])
    Tmin = np.nanmin([np.nanmin(data['IM']), np.nanmin(data['IN']), np.nanmin(data['IF'])])
    
    for key in data:
        if 'I' in key:
            data[key] = np.log10((Tmax-Tmin*1.001)/(data[key]-Tmin*1.001))
            
    # area to mm^2
    s = 38 # p per mm
    # s = 67 # p per mm
    ap = (1/s)**2 * binning**2 if binning else (1/s)**2
    for key in data:
        if 'A' in key:
            data[key] *= ap
            
    # fix nan
    for key in data:
        data[key][np.isnan(data[key])] = np.nanmean(data[key])
            
    # define & apply filters
    def get_filter(signalLength, frameTime, filterTime):
        filterTime = filterTime if filterTime < signalLength*frameTime else signalLength*frameTime/2
        filterTime = int(filterTime/frameTime) if np.mod(int(filterTime/frameTime),2) else int(filterTime/frameTime) + 1
        return filterTime
        
    keys = list(data.keys()).copy()
    for key in keys:
        data[key+'_low1'] = signal.savgol_filter(data[key], get_filter(len(data[key]), frametime, 6*3600), 3)
        data[key+'_low2'] = signal.savgol_filter(data[key], get_filter(len(data[key]), frametime, 1200), 3)
        data[key+'_high1'] = data[key] - data[key+'_low1']
        data[key+'_high2'] = data[key] - data[key+'_low2']
        data[key+'_band'] = data[key+'_low1'] - data[key+'_low2']
        
    # determine range i0..iN where A_F > pt * A_M
    pF = data['AF_low1']/data['AM_low1']
    iF = pF >= pt
    
    m = np.concatenate(( [True], iF==False, [True] ))  # Mask
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits
    
    i0, iN = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits      
    iN = iN - 1  
        
    for key in data:
        data[key] = data[key][i0:iN]
        
    # determine frequencies
    frequencies = {}
    
    t = frametime*np.arange(len(data['AF']))
    for key in data:
        if 'band' in key:
            frequencies[key] = det_frequency(t, data[key], gaplength)
            
    ### region size / thickness figure
    
    # thickness
    plt.figure(figsize=plt.figaspect(1/(1.618)))
    plt.plot(t, data['IM'], 'gray', label='mold')
    plt.plot(t, data['IM_low1'], 'k--')
    plt.plot(t, data['IM_low2'], '--', color='lime')
    plt.plot(t, data['IF'], 'b', label='front')
    plt.plot(t, data['IF_low1'], 'k--')
    plt.plot(t, data['IF_low2'], '--', color='lime')
    plt.plot(t, data['IN'], 'r', label='network')
    plt.plot(t, data['IN_low1'], 'k--', label='filtered (1)')
    plt.plot(t, data['IN_low2'], '--', color='lime', label='filtered (2)')
    
    p, n = crossings(data['IN_band'])
    p, n = fix_gaps(p, n, gaplength)
    
    plt.scatter(t[p], data['IN_low1'][p], marker='^', edgecolors='k', facecolors='none', zorder=np.inf)
    plt.scatter(t[n], data['IN_low1'][n], marker='v', edgecolors='k', facecolors='none', zorder=np.inf)

    p, n = crossings(data['IF_band'])
    p, n = fix_gaps(p, n, gaplength)
    
    plt.scatter(t[p], data['IF_low1'][p], marker='^', edgecolors='k', facecolors='none', zorder=np.inf)
    plt.scatter(t[n], data['IF_low1'][n], marker='v', edgecolors='k', facecolors='none', zorder=np.inf)    
    
    plt.ylabel('thickness (a.u.)',fontsize=14)
    plt.xlabel('time (s)',fontsize=14)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12) 
    plt.legend(fontsize=14, frameon=False)
    
    # area
    plt.figure(figsize=plt.figaspect(1/(1.618)))
    plt.plot(t, data['AM'], 'gray', label='mold')
    plt.plot(t, data['AM_low1'], 'k--')
    plt.plot(t, data['AM_low2'], '--', color='lime')
    plt.plot(t, data['AF'], 'b', label='front')
    plt.plot(t, data['AF_low1'], 'k--')
    plt.plot(t, data['AF_low2'], '--', color='lime')
    plt.plot(t, data['AN'], 'r', label='network')
    plt.plot(t, data['AN_low1'], 'k--', label='filtered (1)')
    plt.plot(t, data['AN_low2'], '--', color='lime', label='filtered (2)')
    
    p, n = crossings(data['AN_band'])
    p, n = fix_gaps(p, n, gaplength)
    
    plt.scatter(t[p], data['AN_low1'][p], marker='^', edgecolors='k', facecolors='none', zorder=np.inf)
    plt.scatter(t[n], data['AN_low1'][n], marker='v', edgecolors='k', facecolors='none', zorder=np.inf)

    p, n = crossings(data['AF_band'])
    p, n = fix_gaps(p, n, gaplength)
    
    plt.scatter(t[p], data['AF_low1'][p], marker='^', edgecolors='k', facecolors='none', zorder=np.inf)
    plt.scatter(t[n], data['AF_low1'][n], marker='v', edgecolors='k', facecolors='none', zorder=np.inf)    
    
    plt.ylabel('area (mm$^2$)',fontsize=14)
    plt.xlabel('time (s)',fontsize=14)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12) 
    plt.legend(fontsize=14, frameon=False)
    
    ### compute and plot correlations
    types = [
            'AN',\
            'AF',\
            'IN',\
            'IF',\
             ]
    
    labels = [
            '$A_N$',\
            '$A_F$',\
            '$I_N$',\
            '$I_F$',\
            ]
    
    p = np.zeros((len(types), len(types)))
    
    for i, Ti in enumerate(types):
        for j, Tj in enumerate(types):
            
            if i >= j:
                p[i, j] = correlate(data[Ti + '_band'], data[Tj + '_band'])
                
            if j >= i:
                p[i, j] = correlate(data[Ti + '_high2'], data[Tj + '_high2'])
            if i==j:
                p[i, j] = np.nan
    
    plt.figure()    
    plt.imshow(p, cmap='bwr', interpolation='none', rasterized=True)
    plt.clim([-1, 1])
    plt.xticks(np.arange(p.shape[0]), labels, fontsize=12)
    plt.yticks(np.arange(p.shape[0]), labels, fontsize=12)
    plt.gca().invert_yaxis()
    
    for j in np.arange(p.shape[0]):
        for i in np.arange(p.shape[0]):
        
            if i != j:
                plt.text(j-0.1,i, '{:.2f}'.format(p[i,j]), fontsize=14)
