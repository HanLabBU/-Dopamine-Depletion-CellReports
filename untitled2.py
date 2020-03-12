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

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']

# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
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

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']

# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
miceList = ['8815']#getMiceList(Files[0]) 

# open all necassary files 
savePath = '/home/dana_z/ssd_2TB/6OHDA/speed2ca/'
df = pd.DataFrame(columns=['mouse','sess','day','period','cre','numred'])
# constents for analysis:
WinPre = 2 #s
WinPost = 2 #s

# for each mouse: 
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

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']

# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
miceList = getMiceList(Files[0]) 

# open all necassary files 
savePath = '/home/dana_z/ssd_2TB/6OHDA/speed2ca/'
df = pd.DataFrame(columns=['mouse','sess','day','period','cre','numred'])
# constents for analysis:
WinPre = 2 #s
WinPost = 2 #s

# for each mouse: 
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

Files = ['FinalData_6OHDA.h5','FinalData_6OHDA_H.h5','FinalData_6OHDA_H_skip.h5','FinalData_6OHDA_skip.h5']

# align spectogram to spike onset -> for each mouse and in total,seperate by CRE:
miceList = ['1222']#getMiceList(Files[0]) 

# open all necassary files 
savePath = '/home/dana_z/ssd_2TB/6OHDA/speed2ca/'
df = pd.DataFrame(columns=['mouse','sess','day','period','cre','numred'])
# constents for analysis:
WinPre = 2 #s
WinPost = 2 #s

# for each mouse: 
for m in miceList:
    data =  getData(Files[0],['speed','trace'],period ='Pre', mice=m)
    cre = getCreType(Files[1],m)
    for sess in tqdm(data.keys()): 
        sess = '8815_day30A'
        if sess[5] == 'B':
            day = 0
        else:
            day = int(re.findall(r'\d+',sess[5:])[0])

        numRed = int(data[sess]['trace']['numred'])


#        if os.path.exists(savePath+'MSN/'+sess):
#            df = df.append({'mouse':m,'sess':sess,'day':day,'period': periodCalc(day),'cre':cre,'numred':numRed},ignore_index=True)
#            continue

         # get data
        Ca = getOnsetOrPeriod(m,sess,'Pre','caOnset_Hf')
        dCa = np.append(Ca[:,1:]-Ca[:,:-1],np.zeros((Ca.shape[0],1)),axis=1)
        dCa[dCa==-1] = 0
        
        speed = data[sess]['speed']['speed']

        # add session to df, so can be retrived
        df= df.append({'mouse':m,'sess':sess,'day':day,'period': periodCalc(day),'cre':cre,'numred':numRed,'numMsn':Ca.shape[0]},ignore_index=True)

        dtS = 1/data[sess]['trace']['FS']
        
        ts = np.arange(0, np.max(data[sess]['trace']['dff'].shape)) * dtS 
        tPlot = np.linspace(-WinPre,WinPost,(WinPre+WinPost)/dtS)      
        

        # for every Cre neuron:
        dca = dCa[0:numRed,:]
        dca = dca[np.sum(dca,axis=1)!=0,:]
        
        for creN in range(0,np.min(dca.shape)):
            onsets = np.full_like(ts,False)
            cN = dca[creN,:]
            for si in ts[cN.astype(bool)]:
                ti = np.argmin(np.abs(ts-si))
                onsets[ti] = True
            al = alignToOnset(speed,(onsets==1), winPost =WinPre/dtS, winPre = WinPost/dtS)

            if al.ndim <2:
                try:
                    al = np.reshape(al,(al.shape[0],1))
                except:
                    print('no onset, when there should be. CRE#',creN,' in sess= ',sess)
                    continue
            al = np.nanmean(al,axis=1,keepdims=True)

            if 'aligned' in locals():
                print(al.shape, aligned.shape)
                aligned = np.concatenate((aligned,al), axis = 1)
            else:
                aligned = al
        
        if np.min(dca.shape)>0:
            if np.isnan(np.sum(aligned)):
                print(sess)
            pickle.dump( aligned, open( savePath+"CRE/"+sess+".pkl", "wb" ) )
            del aligned
        
        # for every MSN neuron:
        dca = dCa[numRed:,:]
        dca = dca[np.sum(dca,axis=1)!=0,:]
        
        for msnN in range(0,np.min(dca.shape)):
            onsets = np.full_like(ts,False)
            mN = dca[msnN,:]
            for si in ts[mN.astype(bool)]:
                ti = np.argmin(np.abs(ts-si))
                onsets[ti] = True
            al = alignToOnset(speed,(onsets==1), winPost =WinPre/dtS, winPre = WinPost/dtS)

            if al.ndim <2:
                try:
                    al = np.reshape(al,(al.shape[0],1))
                except:
                    print('no onset, when there should be. MSN#',msnN,' in sess= ',sess)
                    continue
            if al.shape[1]==0:
                print('no onset, when there should be. MSN#',msnN,' in sess= ',sess)
                continue
            al = np.mean(al,axis=1,keepdims=True)
            if np.isnan(np.sum(al)):
                print('something is wrong')
            if 'aligned' in locals():
                aligned = np.concatenate((aligned,al), axis = 1)
            else:
                aligned = al
        
        if np.min(dca.shape)>0:
            if np.isnan(np.sum(aligned)):
                print(sess)

            pickle.dump( aligned, open( savePath+"MSN/"+sess, "wb" ) )
            del aligned

#df.to_hdf(savePath+'sessions','df')

#
#def getAlignedSpeed(cellType,cre = None, mice = None, period = None, day=None):
#    # function that take in the classification and return the appropreate data:
#    #Inputs:
#    #   cellType - return MSN or CRE if both pass ['MNS','CRE']
#    #   mice - (Optional) list of mice from to include. Default: None - will load data for all mice
#    #   period - (Optional) either 'Pre' or 'Post'. difault: None - return full length of data from picked sessions
#    #   day - (Optional) lambda function with logic for picking days. Default: None - ignore day attr when picking data
#    #           NOTE: day will be ignored if period is specified
#    #   cre - (Optional) which cre mouse is it. options:None (default), "PV", "CHI"
#    #                   must have trace included in dataType list to be taken into account
#    #   WinPre - (Optional) length of pre window in secounds (default 2)
#    #   WinPost - (Optional) length of post window in secounds (default 2)
#    #Output:
#    #   data - the requested data. format: {mice_session:{dataType:data}}
#    
#    
#    dFile = 'FinalData_6OHDA_H.h5'
#    # double check parameters inputs are valid:
#    savePath = '/home/dana_z/ssd_2TB/6OHDA/speed2ca/'
#    df = pd.read_hdf(savePath+'sessions','df')
#    
#    if period == None and day != None and isinstance(day,type(lambda c:None)):
#        df['keep'] = df.apply(lambda row: day(row.day), axis=1)
#        df = df[(df.keep==True)]
#    
#    if period in ['Healthy','Day 1-4','Day 5-12','Day 13-20','One Month']:
#        df = df[(df.period==period)]
#       
#    if cre in ['PV','CHI','NA']:
#        df = df[(df.cre==cre)]
#
#    if not isinstance(cellType,list):
#        cellType = [cellType]
#        
#    cellType = list(set(cellType).intersection(set(['MSN','CRE'])))
#    if len(cellType) == 0:
#        raise ValueError('Not a valid cellType value. cellType must be in ["MSN","CRE"]')
#    
#    # traverse the hdf5 file:
#    if mice == None:
#        mice = getMiceList(dFile) 
#    elif not isinstance(mice,list):
#        mice = [mice]
#    
#    if not isinstance(mice[0],str):
#        for m in range(0,len(mice)):
#            mice[m] = str(mice[m])
#    df = df[(df.mouse.isin(mice))]
#    # start extracting the data:   
#    
#    # alllocate memory:
#    nNeurons = 0;
#    if 'MSN' in cellType:
#        nNeurons = nNeurons + int(df.numMsn.sum()) - int(df.numred.sum())
#    if 'CRE' in cellType:
#        nNeurons = nNeurons + int(df.numred.sum())
#    
#    dResult = np.empty([80,nNeurons],dtype=float)
#    
#    ind = 0
#    for sess in df.sess.unique():
#        if 'MSN' in cellType:
#            try:
#                tempD = pickle.load(open(savePath+'MSN/'+sess,'rb'))
#            except:
#                print('ignored ',sess)
#                continue
##            tempD = np.squeeze(tempD)
##            print(tempD.shape,ind,ind+tempD.shape[1])
#            dResult[:,ind:ind+tempD.shape[1]] = tempD   
#            ind = ind+tempD.shape[1]
#        # for every Cre neuron:
#        if 'CRE' in cellType:
#            try:
#                tempD = pickle.load(open(savePath+'CRE/'+sess,'rb'))
#            except:
#                continue
##            tempD = np.squeeze(tempD)
#            dResult[:,ind:ind+tempD.shape[1]] = tempD   
#            ind = ind+tempD.shape[1]
#        
#    return dResult,df
#
#savePath = '/home/dana_z/ssd_2TB/6OHDA/speed2ca/'
#df = pd.read_hdf(savePath+'sessions','df')
#
#cells = ['MSN']
#figPath = '/home/dana_z/ssd_2TB/6OHDA/figs/speed2CaOnset/'
#tPlot = np.linspace(-2,2,4/0.05)
#for m in df.mouse.unique():
#    for per in df.period.unique():
#        cellType = 'MSN'
#        cre = None
#        A,df2 = getAlignedSpeed(cellType,cre = cre, period = per,mice=m)
#        if len(df2) == 0:
#            continue
#        A = np.round(A,2)
##         if np.nanmax(A)> 1000:
##             print(A[A>100],df2)
#        b = np.mean(A,axis=1)
#        print(b,df2)
#        
##         fig, ax = plt.subplots()
##         ax.plot(tPlot,b)
##         ax.axvline(x=0.0,color='red')
##         ax.set_title(m+' '+per)
##         fig.savefig(figPath+m+'_'+per+'.png',format='png')
##         fig.clf()
##         plt.close(fig)
#        
##         mu = np.mean(b[:int(b.shape[0]/2)],axis=0)
##         Std = np.std(b[:int(b.shape[0]/2)],axis=0)
##         b =(b-mu)/Std
#        
##         fig, ax = plt.subplots()
##         ax.plot(tPlot,b)
##         ax.axvline(x=0.0,color='red')
##         ax.set_title('Z-score'+m+' '+per)
##         fig.savefig(figPath+m+'_'+per+'_Zscore.png',format='png')
##         fig.clf()
##         plt.close(fig)
#
#for per in df.period.unique():
#    cellType = 'MSN'
#    cre = None
#    A,df2 = getAlignedSpeed(cellType,cre = cre, period = per) 
#    A = np.round(A,2)
#    #b = np.nanmean(A,axis=1)
#    #print(np.nanmax(b),np.nanmax(A))
#    print(np.nanmax(A))
##     fig, ax = plt.subplots()
##     ax.plot(tPlot,b)
##     ax.axvline(x=0.0,color='red')
##     ax.set_title('all '+per)
##     fig.savefig(figPath+'all_'+per+'.png',format='png')
##     fig.clf()
##     plt.close(fig)
#
##     mu = np.nanmean(b[:int(b.shape[0]/2)],axis=0)
##     Std = np.nanstd(b[:int(b.shape[0]/2)],axis=0)
##     b =(b-mu)/Std
#
##     fig, ax = plt.subplots()
##     ax.plot(tPlot,b)
##     ax.axvline(x=0.0,color='red')
##     ax.set_title('Z-score all '+per)
##     fig.savefig(figPath+'all_'+per+'_Zscore.png',format='png')
##     fig.clf()
##     plt.close(fig)
