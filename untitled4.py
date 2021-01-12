#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:21:45 2020

@author: dana_z
"""

import os
os.chdir('/home/dana_z/ssd_2TB/6OHDA')
#import mpld3
#mpld3.enable_notebook()
import numpy as np
import scipy as sci
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as Mcolors
import matplotlib.cm as cmx
import sys
import h5py
from IO import *
from utils import *
from plotUtils import *
from ColorSchems import colorPallet as CP
import pptx
from pptx import Presentation 
from pptx.util import Inches
from io import BytesIO
import re
import warnings
import pandas as pd
import sqlalchemy as db
import gc
from tqdm import tqdm
import seaborn as sns
import pywt # wavelet package
import matplotlib.cm as cm

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']
miceList = getMiceList(Files[0])
betaMice = ['8430','4539','7584','7909','1222']

f = h5py.File('Spectograms.hdf5','r')

WinPre = 2 #s
WinPost = 2 #s

periods = {#'Healthy':lambda x: x==0,
           'Day 1-4':lambda x: x<5,
           'Day 5-12':lambda x: (x>4) & (x<13),
           'Day 13-20':lambda x: (x>12) & (x<21),
           'One Month':lambda x: x>21}

sess = '1208_day2'
data =  getData(Files[0],['lfp','trace'],period ='Pre', day=periods['Day 1-4'])
Ca = getOnsetOrPeriod(sess[0:4],sess,'Pre','caOnset_Hf')
dCa = np.append(Ca[:,1:]-Ca[:,:-1],np.zeros((Ca.shape[0],1)),axis=1)
dCa[dCa==-1] = 0

coeff = np.abs(f[m][sess]['Pre']['coeff'].value)
lfpOutliers = removeLFPOutliers(data[sess]['lfp']['lfp'], sess)
try:
    coeff[:,(lfpOutliers[:,0]==1)] = np.nan
    coeff = coeff.T/np.nansum(coeff,axis=1) # So that axis[0] is the time axis + normalize power in frequency per sesion
except:
    print(sess)
#    continue
dtS = float(1/data[sess]['trace']['FS'])
dtL = float(1/data[sess]['lfp']['FS'])
ts = np.arange(0, np.max(data[sess]['trace']['dff'].shape)) * dtS 
tl = np.arange(0, np.max(lfp.shape)) * dtL

tPlot = np.linspace(-WinPre,WinPost,int((WinPre+WinPost)/dtL))     

# for every MSN neuron:
numRed = int(data[sess]['trace']['numred'])
dca = dCa[numRed:,:]
dca = dca[np.sum(dca,axis=1)!=0,:]

for msnN in range(0,1):
    onsetL = np.full_like(tl,False)
    mN = dca[msnN,:]
    for si in ts[mN.astype(bool)]:
        ti = np.argmin(np.abs(tl-si))
        onsetL[ti] = True
#    al = alignToOnset(lfp,(onsetL==1), winPost =WinPre/dtL, winPre = WinPost/dtL)
    al = alignToOnset(coeff,(onsetL==1), winPost =WinPre/dtL, winPre = WinPost/dtL)


#    if al.ndim <2:
#        try:
#            al = np.reshape(al,(al.shape[0],1))
#        except:
#            print('no onset, when there should be. MSN#',msnN,' in sess= ',sess)
#            continue
#
#    if 'aligned' in locals():
#        aligned = np.concatenate((aligned,al), axis = 1)
#    else:
#        aligned = al
    
    if al.ndim <3:
        try:
            al = np.reshape(al,(al.shape[0],al.shape[1],1))
        except:
            print('no onset, when there should be. MSN#',msnN,' in sess= ',sess)
            continue
    
    al = np.nanmean(al,axis=2,keepdims=True)
    al = np.nan_to_num(al,nan=-9999)
    if 'aligned' in locals():
        aligned = np.concatenate((aligned,al), axis = 2)
    else:
        aligned = al
plt.plot(tPlot[:-1],aligned)

plt.pcolormesh(tPlot[:-1],freq,al[:,:,0].T,vmax=5,vmin=-5)

