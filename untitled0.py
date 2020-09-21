#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:14:01 2019

@author: dana_z

"""

def periodCalc(day):
    if day== 0:
        return 'Healthy'
    elif day<5:
        return 'Day 1-4'
    elif day<13:
        return 'Day 5-12'
    elif day<21:
        return 'Day 13-20'
    else:
        return 'One Month'


# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']
miceList = ['1793']

# open all necassary files 
f = h5py.File('Spectograms.hdf5','r') #LFP coeffs
savePath = '/home/dana_z/HD1/lfpAligned2Ca/'
# constents for analysis:
WinPre = 2 #s
WinPost = 2 #s

# for each mouse: 
for m in miceList:
    data =  getData(Files[0],['lfp','trace'],period ='Pre', mice=m,day=lambda x:x==0,drug=b'L-Dopa')
    cre = getCreType(Files[1],m)
    for sess in tqdm(data.keys()): 
        if sess[5] == 'B':
            day = 0
        else:
            day = int(re.findall(r'\d+',sess[5:])[0])

        numRed = int(data[sess]['trace']['numred'])


        if os.path.exists(savePath+'MSN/'+sess):
            df.append({'mouse':m,'sess':sess,'day':day,'period': periodCalc(day),'cre':cre,'numred':numRed},ignore_index=True)
            continue

         # get data
        Ca = getOnsetOrPeriod(m,sess,'Pre','caOnset_Hf')
        dCa = np.append(Ca[:,1:]-Ca[:,:-1],np.zeros((Ca.shape[0],1)),axis=1)
        dCa[dCa==-1] = 0
        
        coeff = np.abs(f[m][sess]['Pre']['coeff'].value)
        lfpOutliers = removeLFPOutliers(data[sess]['lfp']['lfp'], sess)
        try:
            coeff[:,(lfpOutliers[:,0]==1)] = np.nan
            coeff = coeff.T/np.nansum(coeff,axis=1) # So that axis[0] is the time axis + normalize power in frequency per sesion
        except:
            print(sess)
            continue
        
        # add session to df, so can be retrived

        dtS = 1/data[sess]['trace']['FS']
        dtL = 1/data[sess]['lfp']['FS']
        ts = np.arange(0, np.max(data[sess]['trace']['dff'].shape)) * dtS 
        tl = np.arange(0, np.max(data[sess]['lfp']['lfp'].shape)) * dtL

        tPlot = np.linspace(-WinPre,WinPost,(WinPre+WinPost)/dtL)      
        

        # for every Cre neuron:
        dca = dCa[0:numRed,:]
        dca = dca[np.sum(dca,axis=1)!=0,:]
        
        for creN in range(0,np.min(dca.shape)):
            onsetL = np.full_like(tl,False)
            cN = dca[creN,:]
            for si in ts[cN.astype(bool)]:
                ti = np.argmin(np.abs(tl-si))
                onsetL[ti] = True
            al = alignToOnset(coeff,(onsetL==1), winPost =WinPre/dtL, winPre = WinPost/dtL)

            if al.ndim <3:
                try:
                    al = np.reshape(al,(al.shape[0],al.shape[1],1))
                except:
                    print('no onset, when there should be. CRE#',creN,' in sess= ',sess)
                    continue
            al = np.nanmean(al,axis=2,keepdims=True)

            if 'aligned' in locals():
                aligned = np.concatenate((aligned,al), axis = 2)
            else:
                aligned = al
        
        if np.min(dca.shape)>0:
            pickle.dump( aligned, open( savePath+"CRE/"+sess, "wb" ), protocol=4 )
            del aligned
        
        # for every MSN neuron:
        dca = dCa[numRed:,:]
        dca = dca[np.sum(dca,axis=1)!=0,:]
        
        for msnN in range(0,np.min(dca.shape)):
            onsetL = np.full_like(tl,False)
            mN = dca[msnN,:]
            for si in ts[mN.astype(bool)]:
                ti = np.argmin(np.abs(tl-si))
                onsetL[ti] = True
            al = alignToOnset(coeff,(onsetL==1), winPost =WinPre/dtL, winPre = WinPost/dtL)

            if al.ndim <3:
                try:
                    al = np.reshape(al,(al.shape[0],al.shape[1],1))
                except:
                    print('no onset, when there should be. MSN#',msnN,' in sess= ',sess)
                    continue
            
            al = np.mean(al,axis=2,keepdims=True)
            if 'aligned' in locals():
                aligned = np.concatenate((aligned,al), axis = 2)
            else:
                aligned = al
        
        if np.min(dca.shape)>0:
            pickle.dump( aligned, open( savePath+"MSN/"+sess, "wb" ) , protocol=4)
            del aligned

# ===================================================================================
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



# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
# for each mouse: 

#caOnsetW1,caFallW1 = caWideSpikeFinder(dff[:39,:],Fs)

warnings.filterwarnings("ignore")

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']
f = h5py.File('OnsetsAndPeriods.hdf5','r')
# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
miceList = getMiceList(Files[0]) 

# constents for analysis:
tracePerSlide = 8
colors = {'TD':'black','TD_skip':'indianred','MSN':'navy','MSN_skip':'royalblue'}
lf = {'left':0.30, 'top':1.30, 'height':10.80, 'width':25.10}
fArgs = {'left':Inches(lf['left']),'top':Inches(lf['top']), 'height':Inches(lf['height']), 'width':Inches(lf['width'])}

fig, ax = plt.subplots(1,1,figsize=(lf['width'],lf['height']))
fig.set_size_inches(lf['width'],lf['height'],forward=True)

for m in tqdm(miceList):

    Tseconds = 1

    data = getData(Files[1],['trace'],period ='Post', mice=m,drug=b'Saline')
    days = np.zeros(len(data))
    ind = 0
    # sort by session for my own OCD
    for sess in data:
        if sess[5] == 'B':
            day = 0
        else:
            day = int(re.findall(r'\d+',sess[5:])[0])
        days[ind] = day
        ind= ind+1
    a = np.argsort(days)
    dKeys = list(data.keys())
    # calculte high speed period, do 3 sessions per plot, and stor in ppt
    ind = 0;
    for aa in range(0,len(data)):
        sess = dKeys[a[aa]]
        # get traces:
        dff = data[sess]['trace']['dff']
        dt = 1/data[sess]['trace']['FS'][0]
        t = np.linspace(0,dt*dff.shape[1],dff.shape[1])
        Fs = data[sess]['trace']['FS'][0]
        # Vectors are saved as column vectors so.. transposed to raw vector
        if dff.shape[1] == 1:
            dff = dff.T
        # get CaOnset:
        try: 
            caOnset = f[m][sess]['Post']['caOnset_Hf'].value
            caFall =  f[m][sess]['Post']['caFall_Hf'].value
            caOnset[caOnset==0] =np.nan
            caFall[caFall==0] =np.nan
            numred = int(data[sess]['trace']['numred'][0])
            
            tLim = int(600*Fs)
            
            for N in range(0,dff.shape[0]//tracePerSlide+1):
                endN = np.min(((N+1)*tracePerSlide,dff.shape[0]))
                for T in range(0,dff.shape[1]//tLim):
                    endT = np.min(((T+1)*tLim,dff.shape[1]))
                    df = dff[N*tracePerSlide:endN,T*tLim:endT]
                    ca = caOnset[N*tracePerSlide:endN,T*tLim:endT]
                    ca = ca*df
                    cf = caFall[N*tracePerSlide:endN,T*tLim:endT]
                    cf = cf*df
                    spacing = np.max(np.abs(df))
                    Color = ['navy' for x in range(0,df.shape[0])]
                    if tracePerSlide*N <numred:
                        Nl = min(tracePerSlide,numred-N*tracePerSlide)
                        Color[0:Nl] = [colors['TD'] for x in range(0,Nl)]
                        Color[Nl:] = [colors['MSN'] for x in range(Nl,len(Color))]
                    else:
                        Color = [colors['MSN'] for x in range(0,len(Color))]
                    for d in range(0,df.shape[0]):
                        ax.plot(t[T*tLim:endT],df[d,:]+d*spacing,color=Color[d]) 
                        ax.plot(t[T*tLim:endT],ca[d,:]+d*spacing,color='red')
                        ax.plot(t[T*tLim:endT],cf[d,:]+d*spacing,color='gold')
                    fig.savefig('/home/dana_z/HD1/6OHAD_figures/Post_Saline/'+sess+'_'+str(N)+'_'+str(T)+'.png',transparent=False,format='png')
                    ax.cla()
#                    plt.close(fig)
        except:
            print(sess,' error')
            continue