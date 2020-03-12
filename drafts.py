#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:18:26 2019

@author: dana_z
"""

def PlotRelativeToOnset2(ax,aligned,tPlot,Color='black',Label='',
                        linewidth=3,orizColor='darkred',orizLine=0.0,
                        orizStyle='dashed', Alpha = 0.1,mesErr = False):
    #this function takes the alligned data and plot it on plt axis 'ax'
    #TODO: sense checks on parameters... + hundle no oriz line
    #TODO: add paramas for legend and titles etc. 
    d = np.nanmean(aligned,axis=1)
    sd = np.nanstd(aligned,axis=1)
    if mesErr:
        sd = sd/np.sqrt(aligned.shape[1])
    
    ax.plot(tPlot,d,linewidth=3,color=Color,label=Label)
    for l in range(0,aligned.shape[1]):
        ax.plot(tPlot,aligned[:,l],linewidth=3,color=Color,alpha=Alpha)
    #ax.fill_between(tPlot, d-sd, d+sd,color=Color,alpha=Alpha)
    #ax.axvline(x=orizLine,color=orizColor,linestyle=orizStyle)



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


Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']
miceList = getMiceList(Files[0])

#data = getData(Files[0],['speed','trace'],period ='Pre', mice='1208',day = lambda x: x==0,drug=b'Amphetamin')

#sess =list(data.keys())[0]
#speed = data[sess]['speed']['speed'].T
#dff = data[sess]['trace']['dff']
        # Vectors are saved as column vectors so.. transposed to raw vector
#Fs = data[sess]['trace']['FS'][0]

tracePerSlide = 8
colors = {'TD':'black','TD_skip':'indianred','MSN':'navy','MSN_skip':'royalblue'}
lf = {'left':0.30, 'top':1.30, 'height':10.80, 'width':25.10}
fArgs = {'left':Inches(lf['left']),'top':Inches(lf['top']), 'height':Inches(lf['height']), 'width':Inches(lf['width'])}

fig, ax = plt.subplots(1,1,figsize=(lf['width'],lf['height']))
fig.set_size_inches(lf['width'],lf['height'],forward=True)



tapers = [2,3]
std_threshold = 4
window_size = 1
pre_window = 10

roi_list = {}
for m in tqdm(miceList):

    Tseconds = 1

    data = getData(Files[1],['trace'],period ='Pre', mice=m)
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
        Fs = data[sess]['trace']['FS'][0]
        caOnset = np.full_like(dff,np.nan)
        x_axis = np.arange(0,dff.shape[1])/20

        for roi in range(0,dff.shape[0]):
            print('ROI ',str(roi),'/',str(dff.shape[0]))
            whole_trace = dff[roi,:]
            S,t,f,_=mtspecgramc(whole_trace,[window_size, 1/Fs],tapers=tapers,Fs=Fs)
            normalized_S = S-np.mean(S,axis=0)
            normalized_S = np.divide(normalized_S,np.std(S,axis=0,ddof=1))
            
            func = lambda x: (x >= 0) and (x<=fpass[-1])
            f_idx = [i for (i, val) in enumerate(f) if (val>=0 and val <=2)]
        
            power = np.mean(normalized_S[:,f_idx],axis=1)
            d_power= power[1:]-power[:-1]
            
            scaleMedian = 3*1.4826*np.median(np.abs(d_power-np.median(d_power))) 
            up_power_idx_list = [i for (i, val) in enumerate(d_power) if val>scaleMedian]
            if len(up_power_idx_list) == 0:
                continue
            up_power_idx_list = [up_power_idx_list[0]]+[val for (i, val) in enumerate(up_power_idx_list) if val-up_power_idx_list[i-1]>1]
            up_power_idx_list = [val for (i, val) in enumerate(up_power_idx_list) if d_power[val]>np.mean(d_power)]
        
            down_power_idx_list =[d_power.size for (i, val) in enumerate(up_power_idx_list)] 
                
            for idx,up_power_idx in enumerate(up_power_idx_list):
                current_d_power = d_power[up_power_idx:]
                try:
                    down_power_idx_list[idx] = up_power_idx+ np.min([i for (i,val) in enumerate(current_d_power) if val<=0])
                except:
                    down_power_idx_list[idx] = up_power_idx
        
            event_time = np.asarray([t[up_power_idx_list] , t[down_power_idx_list]]).T
            event_idx = np.full_like(event_time,np.nan)
            event_amp =  np.full((event_time.shape[0],1),np.nan)
            pre_event_threshold = np.full((event_time.shape[0],1),np.nan)
        
            for spike_time_idx,eTime in enumerate(event_time):
                start_idx = np.argmin(np.abs(x_axis-eTime[0]))
                end_idx = np.argmin(np.abs(x_axis-eTime[1]))
        
                current_trace = whole_trace[end_idx:]
                d_current_trace = current_trace[1:]-current_trace[:-1]
                extra_end_idx = [i for (i, val) in enumerate(d_current_trace) if val<=0]
                if len(extra_end_idx)>2:
                    extra_end_idx = extra_end_idx[:2]
                    
                try:
                    end_idx = end_idx+extra_end_idx[-1]
                except:
                    print('skipped iteration')
                    continue
                   
                current_trace = whole_trace[start_idx:end_idx+1]
                ref_idx = start_idx-1
                max_amp = np.max(current_trace)
                max_idx = np.argmax(current_trace)
                end_idx = ref_idx+max_idx
                if max_idx>0:
                    min_amp = np.min(current_trace[:max_idx+1])
                    min_idx = np.argmin(current_trace[:max_idx+1])
                else:
                    min_amp = current_trace[0]
                    min_idx = 0
                start_idx = ref_idx+min_idx
        
                pre_start_idx = np.max([0,int(start_idx-pre_window*Fs)])
                pre_event_threshold[spike_time_idx] = std_threshold*np.std(whole_trace[pre_start_idx:start_idx+1],ddof=1)
                event_amp[spike_time_idx] = max_amp-min_amp
                event_time[spike_time_idx,0] = x_axis[start_idx]
                event_time[spike_time_idx,1] = x_axis[end_idx]
                event_idx[spike_time_idx,0] = start_idx
                event_idx[spike_time_idx,1] = end_idx

        
            pre_event_threshold = np.delete(pre_event_threshold,np.nonzero(np.isnan(event_amp))[0])
            event_time = np.delete(event_time,np.nonzero(np.isnan(event_amp))[0], axis=0)
            event_idx = np.delete(event_idx,np.nonzero(np.isnan(event_amp))[0], axis=0)
            event_amp = np.delete(event_amp,np.nonzero(np.isnan(event_amp))[0], axis=0)
               
            event_time = np.delete(event_time,np.where(event_amp[:,0]<pre_event_threshold),axis=0)
            event_idx = np.delete(event_idx,np.where(event_amp[:,0]<pre_event_threshold),axis=0)
            event_amp = np.delete(event_amp,np.where(event_amp[:,0]<pre_event_threshold),axis=0)
        
#            roi_list[roi] ={'event_time':event_time,'event_idx':event_idx,'event_amp':event_amp}
            for st, en in event_idx:
                caOnset[roi,int(st):int(en)+1] = dff[roi,int(st):int(en)+1]
        
        numred = int(data[sess]['trace']['numred'][0])
        for N in range(0,dff.shape[0]//tracePerSlide):
            df = dff[N*tracePerSlide:tracePerSlide*N+tracePerSlide,:]
            ca = caOnset[N*tracePerSlide:tracePerSlide*N+tracePerSlide,:]
            spacing = np.max(df)
            Color = ['navy' for x in range(0,df.shape[0])]
            if tracePerSlide*N <numred:
                Nl = min(tracePerSlide,numred-N*tracePerSlide)
                Color[0:Nl] = [colors['TD'] for x in range(0,Nl)]
                Color[Nl:] = [colors['MSN'] for x in range(Nl,len(Color))]
            else:
                Color = [colors['MSN'] for x in range(0,len(Color))]
            for d in range(0,df.shape[0]):
                ax.plot(x_axis,df[d,:]+d*spacing,color=Color[d]) 
                ax.plot(x_axis,ca[d,:]+d*spacing,color='red')
            fig.savefig('/home/dana_z/HD1/6OHAD_figures/'+sess+'_'+str(N)+'.png',transparent=False,format='png')
            ax.cla()
#    return roi_list
#trace_event_detection.m
#Displaying trace_event_detection.m.    
            
            
            def getLambdaFromPeriod(period):
    conver = {'Healthy':lambda day: day== 0,
              'Day 1-4': lambda day: (day<5)&(day>0),
              'Day 5-12': lambda day: (day<13)&(day>=5),
              'day 13-20': lambda day: (day<21)&(day>=13),
              'One Month': lambda day: day>=21}
    try:
        return conver[period]
    except:
        return None

    

def getAlignedLFP(cellType,cre = None, mice = None, period = None, day=None,WinPre=2,WinPost=2):
    # function that take in the classification and return the appropreate data:
    #Inputs:
    #   cellType - return MSN or CRE if both pass ['MNS','CRE']
    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
    #           NOTE: day will be ignored if period is specified
    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
    #                   must have trace included in dataType list to be taken into account
    #   WinPre - (Optional) length of pre window in secounds (default 2)
    #   WinPost - (Optional) length of post window in secounds (default 2)
    #Output:
    #   data - the requested data. format: {mice_session:{dataType:data}}
    
    

    # double check parameters inputs are valid:
    
    if day != None and not isinstance(day,type(lambda c:None)):
        day = None
        warnings.warn('Invalid input. day must be a lambda function')
    
    if period not in [None,'Healthy','Day 1-4','Day 5-12','Day 13-20','One Month']:
        period = None
        warnings.warn('Invalid input. Period must be in [Healthy,Day 1-4,Day 5-12,Day 13-20,One Month].')
    
    if period is not None and isinstance(period,str):
        day = getLambdaFromPeriod(period)
        
    if cre not in [None,'PV','CHI','NA']:
        cre = None
        warnings.warn('Invalid input. Cre must be either "PV" or "CHI".')

    if not isinstance(cellType,list):
        cellType = [cellType]
        
    cellType = list(set(cellType).intersection(set(['MSN','CRE'])))
    if len(cellType) == 0:
        raise ValueError('Not a valid cellType value. cellType must be in ["MSN","CRE"]')
        
    f = h5py.File('Spectograms.hdf5','r')
    dFile = 'FinalData_6OHDA_H.h5'
    # traverse the hdf5 file:
    if mice == None:
        mice = getMiceList(dFile) 
    elif not isinstance(mice,list):
        mice = [mice]
    
    if not isinstance(mice[0],str):
        for m in range(0,len(mice)):
            mice[m] = str(mice[m])
        
    # start extracting the data:      
    data =  getData(dFile,['lfp','trace'],period ='Pre', mice=mice, day=day,cre=cre)
    for sess in data.keys():
        m = sess[0:4]
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
        numRed = int(data[sess]['trace']['numred'])
        dtS = 1/data[sess]['trace']['FS']
        dtL = 1/data[sess]['lfp']['FS']
        ts = np.arange(0, np.max(data[sess]['trace']['dff'].shape)) * dtS 
        tl = np.arange(0, np.max(data[sess]['lfp']['lfp'].shape)) * dtL

        tPlot = np.linspace(-WinPre,WinPost,(WinPre+WinPost)/dtL)      
             
        # for every Cre neuron:
        if 'CRE' in cellType:
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
    
                if 'creAligned' in locals():
                    creAligned = np.concatenate((creAligned,al), axis = 2)
                else:
                    creAligned = al
        
        if 'MSN' in cellType:
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
                
                al = np.nanmean(al,axis=2,keepdims=True)
                if 'MSNaligned' in locals():
                    MSNaligned = np.concatenate((MSNaligned,al), axis = 2)
                else:
                    MSNaligned = al
            
    Rdata = {'numSess':len(data.keys())}
    if 'MSN' in cellType and 'MSNaligned' in locals():
        Rdata['MSN'] = MSNaligned
    if 'CRE' in cellType and 'creAligned' in locals():
        Rdata['CRE'] = creAligned
    
    
    return Rdata
    
A = getAlignedLFP('MSN',period = 'Healthy')