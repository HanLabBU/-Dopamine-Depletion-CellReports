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

periods = {'Healthy': lambda day: day== 0,
           'Day 1-4': lambda day: (day >0)&(day<5),
           'Day 5-12': lambda day: (day >4)&(day<13),
           'Day 13-20':lambda day: (day >12)&(day<21),
           'One Month': lambda day: day > 20}

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']

# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
miceList = ['1222']#getMiceList(Files[0]) 

# open all necassary files 
savePath = '/home/dana_z/ssd_2TB/6OHDA/figs/rawLFP2Ca/'
df = pd.DataFrame(columns=['mouse','sess','day','period','cre','numred'])
# constents for analysis:
WinPre = 2 #s
WinPost = 2 #s

# for each mouse: 
for per in periods.keys(): # m in miceList:
    data =  getData(Files[0],['lfp','trace'],period ='Pre', day = periods[per])
#    cre = getCreType(Files[1],m)
    for sess in tqdm(data.keys()): 
        m = sess[:4]
        if sess[5] == 'B':
            day = 0
        else:
            day = int(re.findall(r'\d+',sess[5:])[0])

        numRed = int(data[sess]['trace']['numred'])


         # get data
        Ca = getOnsetOrPeriod(m,sess,'Pre','caOnset_Hf')
        dCa = np.append(Ca[:,1:]-Ca[:,:-1],np.zeros((Ca.shape[0],1)),axis=1)
        dCa[dCa==-1] = 0
        
        lfp = data[sess]['lfp']['lfp']
        lfp = lfp/np.sum(lfp)
        lfpOutliers = removeLFPOutliers(lfp, sess)
        try:
            lfp[(lfpOutliers[:,0]==1)] = np.nan   
        except:
            print(sess)
            continue
        
        # add session to df, so can be retrived
#        df= df.append({'mouse':m,'sess':sess,'day':day,'period': periodCalc(day),'cre':cre,'numred':numRed,'numMsn':Ca.shape[0]},ignore_index=True)

        dtS = float(1/data[sess]['trace']['FS'])
        dtL = float(1/data[sess]['lfp']['FS'])
        ts = np.arange(0, np.max(data[sess]['trace']['dff'].shape)) * dtS 
        tl = np.arange(0, np.max(data[sess]['lfp']['lfp'].shape)) * dtL

        tPlot = np.linspace(-WinPre,WinPost,int((WinPre+WinPost)/dtL))     
        

        # for every Cre neuron:
        dca = dCa[0:numRed,:]
        dca = dca[np.sum(dca,axis=1)!=0,:]
        
#        for creN in range(0,np.min(dca.shape)):
#            onsetL = np.full_like(tl,False)
#            cN = dca[creN,:]
#            for si in ts[cN.astype(bool)]:
#                ti = np.argmin(np.abs(tl-si))
#                onsetL[ti] = True
#            al = alignToOnset(lfp,(onsetL==1), winPost =WinPre/dtL, winPre = WinPost/dtL)
#
#            if al.ndim <3:
#                try:
#                    al = np.reshape(al,(al.shape[0],al.shape[1],1))
#                except:
#                    print('no onset, when there should be. CRE#',creN,' in sess= ',sess)
#                    continue
#            al = np.nanmean(al,axis=2,keepdims=True)
#            al = np.nan_to_num(al,nan=-9999)
#            
#            if 'aligned' in locals():
#                aligned = np.concatenate((aligned,al), axis = 2)
#            else:
#                aligned = al
#        
#        if np.min(dca.shape)>0:
#            pickle.dump( aligned, open( savePath+"CRE/"+sess, "wb" ), protocol=4 )
#            del aligned
        
        # for every MSN neuron:
        dca = dCa[numRed:,:]
        dca = dca[np.sum(dca,axis=1)!=0,:]
        
        for msnN in range(0,np.min(dca.shape)):
            onsetL = np.full_like(tl,False)
            mN = dca[msnN,:]
            for si in ts[mN.astype(bool)]:
                ti = np.argmin(np.abs(tl-si))
                onsetL[ti] = True
            al = alignToOnset(lfp,(onsetL==1), winPost =WinPre/dtL, winPre = WinPost/dtL)

            if al.ndim <2:
                try:
                    al = np.reshape(al,(al.shape[0],1))
                except:
                    print('no onset, when there should be. MSN#',msnN,' in sess= ',sess)
                    continue
            
            al = np.nanmean(al,axis=1,keepdims=True)
            al = np.nan_to_num(al,nan=9999.0)
            if 'aligned2' in locals():
                aligned2 = np.concatenate((aligned2,al), axis = 1)
            else:
                aligned2 = al
        try:    
            aligned2 = np.nanmean(aligned2,axis=1,keepdims=True)
        except:
            print(sess)
            continue
        if 'aligned' in locals():
            aligned = np.concatenate((aligned2,aligned), axis = 1)
        else:
            aligned = aligned2

    
    # create and save figure for period:
    aligned[aligned == 9999.0] = np.nan
#    fig, ax = plt.subplots(2,1)
#    ax[0].plot(tPlot[:-1],aligned,alpha=0.1,color='blue')
#    ax[0].axvline(x=0.0,color='red')
#    ax[1].plot(tPlot[:-1],aligned,alpha=0.1,color='blue')
#    ax[1].axvline(x=0.0,color='red')
#    ax[1].set_xlim(-.5,.5)
#    ax[0].set_title(per+' (Raw signal)')
#    
#    fig.savefig(savePath+per+'.png',format='png')
#    fig.clf()
#    plt.close(fig)
    
    mu = np.nanmean(aligned[:int(aligned.shape[0]/2),:],axis=0)
    Std = np.nanstd(aligned[:int(aligned.shape[0]/2),:],axis=0)
    aligned =(aligned-mu)/Std
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(tPlot[:-1],aligned,alpha=0.1,color='blue')
    ax[0].plot(tPlot[:-1],np.mean(aligned,axis=1),color='black')
    ax[0].axvline(x=0.0,color='red')
    ax[1].plot(tPlot[:-1],aligned,alpha=0.1,color='blue')
    ax[1].plot(tPlot[:-1],np.mean(aligned,axis=1),color='black')
    ax[1].axvline(x=0.0,color='red')
    ax[1].set_xlim(-.5,.5)
    ax[0].set_title(per+' (Z-score signal)')
    ax[0].set_ylim(-4,4)
    ax[1].set_ylim(-4,4)
    
    fig.savefig(savePath+per+'_Zscore.png',format='png')
    fig.clf()
    plt.close(fig)
    
#    fig, ax = plt.subplots(2,1)
#    ax[0].plot(tPlot[:-1],np.mean(aligned,axis=1),alpha=0.1,color='blue')
#    ax[0].axvline(x=0.0,color='red')
#    ax[1].plot(tPlot[:-1],aligned,alpha=0.1,color='blue')
#    ax[1].axvline(x=0.0,color='red')
#    ax[1].set_xlim(-.5,.5)
#    ax[0].set_title(per+' (Z-score signal)')
#    ax[0].set_ylim(-4,4)
#    ax[1].set_ylim(-4,4)
    
    fig.savefig(savePath+per+'_Zscore_mean.png',format='png')
    fig.clf()
    plt.close(fig)
    
    del aligned
        

   
        
#        if np.min(dca.shape)>0:
#            pickle.dump( aligned, open( savePath+"MSN/"+sess, "wb" ) , protocol=4)
#            del aligned



