#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:00:37 2020

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
from chronux import *
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
from scipy.stats.distributions import chi2
from numpy.lib.stride_tricks import as_strided
import pickle



warnings.filterwarnings("ignore")

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']

# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
miceList = getMiceList(Files[1]) 

# constents for analysis:
WinPre = 2 #s
WinPost = 2 #s

# for each mouse: 
lfpPath = '/home/dana_z/HD1/lfpAligned2Ca/'

figPath = '/home/dana_z/ssd_2TB/6OHDA/figs/paper1_edit3/'
df = pd.read_csv(lfpPath+'sessions')
f = h5py.File('Spectograms.hdf5','r')
freq = f['0761']['freq'].value

f.close()

cells = ['MSN']
dtL = 0.00032768
tPlot = np.linspace(-2,2,int(4/dtL)-1)

data = {'Healthy':{},'Day 1-4':{},'Day 5-12':{},'Day 13-20':{},'One Month':{}}
numIter = 1000

# get the data per mouse per period
for m in miceList:
    for per in df.period.unique():
        cellType = 'CRE'
        cre = 'CHI'
        A,df2 = getAlignedLFP(cellType,cre = cre, period = per,mice=m)
        if len(df2) == 0:
            continue
        A[A==9999.0] = np.nan
        A[A==-9999.0] = np.nan    
        if A.shape[2] > 1000:
            b = A[:,:,0:1000]
            mu = np.mean(b[:int(b.shape[0]/2),:,:],axis=0)
            Std = np.std(b[:int(b.shape[0]/2),:,:],axis=0)
            b =(b-mu)/Std
            b = np.nansum(b,axis=2)
            for ind in range(0,A.shape[2]//1000):
                c = A[:,:,1000*(ind+1):np.min([A.shape[2],1000*(ind+2)])]
                mu = np.nanmean(c[:int(c.shape[0]/2),:,:],axis=0)
                Std = np.nanstd(c[:int(c.shape[0]/2),:,:],axis=0)
                c =(c-mu)/Std
                c = np.nansum(c,axis=2)
                b = b+c
            b= b/A.shape[2]
        else:
            b = A
            mu = np.nanmean(b[:int(b.shape[0]/2),:,:],axis=0)
            Std = np.nanstd(b[:int(b.shape[0]/2),:,:],axis=0)
            b =(b-mu)/Std
            b = np.nanmean(b,axis=2)

        data[per][m] = b
        del A,b

#get the substructed data in a format that we can actually work with:
norData = {}
distData = {}
x,y = data['Healthy']['0761'].shape
t200 = (tPlot >0) & (tPlot<0.2)
gamma = freq >= 10    
for per in data.keys():
    if per == 'Healthy':
        continue
    mice = list(data[per].keys())
    nMice = len(mice)
    norData[per] = np.empty((x,y,nMice))
    for mInd in range(0,nMice): 
        m = mice[mInd]
        norData[per][:,:,mInd] = data[per][m]-data['Healthy'][m]
    
    # create a distribution by choching mice at random 1000 times:
    inds = np.random.randint(nMice-1,size = (nMice,numIter))
    distData[per] = np.empty((np.sum(gamma),numIter))
    for itr in range(0,numIter):
        temp  = norData[per][:,:,inds[:,itr]]
        temp = np.nanmean(temp[t200,:,:][:,gamma,:],axis=0)
        distData[per][:,itr] = np.nanmean(temp,axis=1)
        if itr%100 == 0:
            print('finishd ',itr, ' iterations ... ')
     
    # plot histograms 
    mu = np.mean(distData[per],axis = 0)
#    plt.hist(mu)
    print(np.percentile(mu,[2.5,97.5]))
        

# pot and save figures... 
#for per in data.keys():
#    if per == 'Healthy':
#        continue
#    
#    fig, ax = plt.subplots(1,1,figsize=(20,20))
#    ax.boxplot(distData[per].T,whis= [2.5,97.5],flierprops={'markersize':3, 'alpha':0.5})
#    ax.set_xticklabels(np.round(freq[gamma],2) , rotation=45, ha='right')
#    ax.axhline(0,color='red')
#    ax.set_title(per + ' - Bootstrap confident intervals')
#    fig.savefig(figPath+per+'_BootstrapIntervals_2_0s_NoBetaMice.png',format='png')
#    fig.clf()
#    plt.close(fig)
    
# Average high gamma and see if works?
    


distData2 = {}
x,y = data['Healthy']['0761'].shape
t200 = (tPlot >0.0) & (tPlot<0.2)
fr = {'high_Gamma':freq >= 60, 'low_Gamma':(freq >= 40) & (freq<60), 'high_Beta_15-20':(freq >= 15) & (freq<20),
     'Beta_10-15':(freq >= 10) & (freq<15)}
lFr = len(fr.keys())
     
for per in data.keys():
    if per == 'Healthy':
        continue
    inds = np.random.randint(nMice-1,size = (nMice,numIter))
    distData2[per] = np.empty((lFr,numIter))
    for itr in range(0,numIter):
        for ind,k in enumerate(fr.keys()):
            highGamma = fr[k]
            temp  = norData[per][:,:,inds[:,itr]]
            temp = np.nanmean(temp[t200,:,:][:,highGamma,:],axis=0)
            temp = np.nanmean(temp,axis=0)
            distData2[per][ind,itr] = np.nanmean(temp,axis=0)
        if itr%100 == 0:
            print('finishd ',itr, ' iterations ... ')

    fig, ax = plt.subplots(1,1,figsize=(20,20))
    ax.boxplot(distData2[per][~np.isnan(distData2[per])].T,whis= [2.5,97.5],flierprops={'markersize':1, 'alpha':0.5},
               boxprops={'linewidth':2},whiskerprops={'linewidth':2},
               medianprops ={'linewidth':2} )
    ax.set_xticklabels(fr.keys() , rotation=45, ha='right',fontsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.axhline(0,color='red')
    ax.set_title(per + ' - Bootstrap confident intervals',fontsize=20)
    fig.savefig(figPath+per+'_BootstrapIntervals_CHI_200ms.png',format='png')
    fig.savefig(figPath+per+'_BootstrapIntervals_CHI_200ms.svg',format='svg')
    fig.clf()
    plt.close(fig)



def setBoxColors(bp):
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    #setp(bp['medians'][0], color='blue')
    
fig, ax = plt.subplots(1,1,figsize=(20,20))
posInd = 1;
labs = [list(fr.keys())[0],list(fr.keys())[3]]
for per in data.keys():
    if per == 'Healthy':
        continue
    bp = ax.boxplot(distData2[per][[0,3],:].T,positions = [posInd,posInd+1],whis= [2.5,97.5],flierprops={'markersize':1, 'alpha':0.5},
               boxprops={'linewidth':2},whiskerprops={'linewidth':2},
               medianprops ={'linewidth':2} )
    posInd = posInd+3
    setBoxColors(bp)

ax.set_xticklabels(list(data.keys())[1:], rotation=45, ha='right',fontsize=20) 
ax.set_xticks([1.5, 4.5, 7.5,10.5])   
ax.tick_params(axis='y', which='major', labelsize=20)
ax.axhline(0,color='red')
ax.set_title('Bootstrap confident intervals',fontsize=20)
hB, = plt.plot([1,1],'b-')
hR, = plt.plot([1,1],'k-')
plt.legend((hB, hR),(labs[0], labs[1]))
hB.set_visible(False)
hR.set_visible(False)

fig.savefig(figPath+'BootstrapIntervals_CHI_200ms_all.png',format='png')
fig.savefig(figPath+'BootstrapIntervals_CHI_200ms_all.svg',format='svg')

fig.clf()
plt.close(fig)