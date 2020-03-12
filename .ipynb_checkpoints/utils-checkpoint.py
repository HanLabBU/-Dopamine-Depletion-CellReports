# utility function meant specifically for the 6OHDA project
# written by Dana Zemel
# last edited 10/9/2018 (most code originally written nov. 2017)

import numpy as np

def getPowerInBand(f,Sxx,minf, maxf):
    # get the mean power in a frequency band:
    # input: 
    #     f - frequencies of spectogram
    #     Sxx - power output of spectogram
    #     fmin - lower lim of froqency band
    #     fmax - upper lim of frequency band
    
    # TODO: add sense check on variables
    ind = indices(f, lambda x: x >= minf and x<=maxf)
    return np.sum(Sxx[ind,:],0) 
    

def indices(a, func):
    # matlab equvalent of find from: 
    #   https://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python
    return [i for (i, val) in enumerate(a) if func(val)]

def getPowerSpec(lfpDict):
    # This function takes in a dict with lfp data (that was returned from getData())
    # and returns the average power spectra
    # Inputs: 
    #   lfpDict - a dictionary with lfp data as returned from getData()
    # Outputs:
    #   M - mean power spectra
    #   Std - standard diviation of power spectra
    #   f - frequency list

    data = []
    for j in lfpDict:
        lfp = lfpDict[j]['lfp']['lfp']
        f, t, Sxx = signal.spectrogram(lfp[0,:],lfpDict[j]['lfp']['FS'],window=('hamming'),nperseg=140,noverlap =120,nfft=1200)

        Power = np.sum(Sxx,1)
        totPower = np.sum(Power)
        #beta = np.mean(getPowerInBand(f,Sxx,13,20)/np.sum(Sxx,axis = 0)
        data.append(Power/totPower)    



    data = np.array(data)
    M = np.mean(data,axis=0)
    Std = np.std(data, axis = 0)
    return M, Std, f