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



Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']
miceList = getMiceList(Files[0])
betaMice = ['8430','4539','7584','7909','1222']

#path to folder where figures should be saved
figFolder = '/home/dana_z/ssd_2TB/6OHDA/figs/paper1_edit2/'

#  for all mice - by period
#PV0 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='PV',red = True,day = lambda x: x==0)
#PV1 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='PV',red = True,day = lambda x: (x>0)& (x<5) )
#PV2 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='PV',red = True,day = lambda x: (x>4)& (x<13) )
#PV3 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='PV',red = True,day = lambda x: (x>12)& (x<21) )
#PV4 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='PV',red = True,day = lambda x: x>20 )
#
#CHI0 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='CHI',red = True,day = lambda x: x==0)
#CHI1 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='CHI',red = True,day = lambda x: (x>0)& (x<5) )
#CHI2 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='CHI',red = True,day = lambda x: (x>4)& (x<13) )
#CHI3 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='CHI',red = True,day = lambda x: (x>12)& (x<21) )
#CHI4 = getData(Files[1],['speed','rot','trace'],period ='Pre', cre='CHI',red = True,day = lambda x: x>20 )
#
#MSN0 = getData(Files[1],['speed','rot','trace'],period ='Pre', red = False,day = lambda x: x==0)
#MSN1 = getData(Files[1],['speed','rot','trace'],period ='Pre', red = False,day = lambda x: (x>0)& (x<5) )
#MSN2 = getData(Files[1],['speed','rot','trace'],period ='Pre', red = False,day = lambda x: (x>4)& (x<13) )
#MSN3 = getData(Files[1],['speed','rot','trace'],period ='Pre', red = False,day = lambda x: (x>12)& (x<21) )
#MSN4 = getData(Files[1],['speed','rot','trace'],period ='Pre', red = False,day = lambda x: x>20 )

# chol/PV/MSN triggered MSNs 
fig, ax = plt.subplots(4,5,figsize=(20, 5),sharex='col',sharey='row')
Colors = CP('creType')

#cond = ['PVdataH','PVdataP','MSNdataH','MSNdataP','CHIdataH','CHIdataP']
cond = ['PV','CHI','MSN']
Period = {'Healthy':0,'Day 1-4':1,'Day 5-12':2,'Day 13-20':3,'One Month':4}

# set the time range to plot: (Assuming all data is in 20Hz, if dataset changes, change this!)
preS = 80 #2s
PostS = 80 # 2s
dt = 0.05

tPlot = np.linspace(-preS*dt,PostS*dt,preS+PostS)
quant = []

for p in tqdm(Period.keys()):
    axInd = 0;
    for c in cond:
        data = eval(c+str(Period[p]))
        for s in data.keys():
            m = s[:4]
            Ca = getOnsetOrPeriod(m,s,'Pre','caOnset_Hf')
            dCa = Ca[:,1:]-Ca[:,:-1]
            dCa[dCa==-1] = 0;
            Zdff = np.sum(dCa,axis=0)
            speed = data[s]['speed']['speed'].T
            # calc MSNs
            sOnset = getOnsetOrPeriod(m,s,'Pre','mvmtOnset')        
            sA = alignToOnset(Zdff, sOnset, winPost=PostS,winPre=preS)  
            if Zdff.shape[1] ==1 and np.sum(sOnset)>0:

                try:
                    sA = np.reshape(sA,(sA.shape[0],1,sA.shape[1]))
                except:
                    sA = np.reshape(sA,(sA.shape[0],1,1))

            if len(sA.shape) > 2:
                sA = np.mean(sA,2)
                if 'sAligned' not in locals():
    #               print(s+' :',sA.shape)
                    sAligned = sA
                else:
    #                print(s+' :',sA.shape,sAligned.shape)
                    sAligned = np.concatenate((sAligned,sA),axis=1)

            if c == 'MSN':
                sS = alignToOnset(speed, sOnset, winPost=PostS,winPre=preS)
                if sS.ndim > 1:
                    if 'sAlignedS' not in locals():
        #           print(s+' :',sA.shape)
                        sAlignedS = sS
                    else:
    #                print(s+' :',cAs.shape,caAlignedS.shape)
                        sAlignedS = np.concatenate((sAlignedS,sS),axis=1) 
        
        quant.append({'pre':np.nanmean(sAligned[:preS+1,:],axis=0),
                      'post':np.nanmean(sAligned[preS+1:,:],axis=0),'cre':c,'period':p})
#        sAligned = sAligned-np.mean(sAligned[:preS+1,:],axis=0)
#        ax[axInd,Period[p]].hist(tPlot,bins = tPlot,weights = np.sum(sAligned,axis=1)/np.sum(sAligned),
#                                Color=Colors[c],Label=c)
        sns.histplot(x=tPlot,weights=np.sum(sAligned,axis=1)/np.sum(sAligned),ax=ax[axInd,Period[p]],color=Colors[c],kde=True,bins=len(tPlot))
        print(np.max(np.sum(sAligned,axis=1))/np.sum(sAligned))
        sw = sAligned
        axInd = axInd+1;
        if c== 'MSN':
            PlotRelativeToOnset(ax[3,Period[p]],sAlignedS,tPlot,Color='black',Label='speed',mesErr=True)
#            ax[3,Period[p]].plot(tPlot,sAlignedS,linewidth=3,color='black',label='speed',alpha = .2)
            del sAlignedS
        del sAligned
    ax[0,Period[p]].set_title(p)
    ax[0,Period[p]].legend(fontsize=10)
    ax[1,Period[p]].legend(fontsize=10)
ax[1,0].set_title('speed')

fig.savefig(figFolder+'cells_aligned_to_Speed_hist.png',transparent=True,format='png')
# #fig2, ax2 = plt.subplots(1,2,sharex='col',sharey='row')
