from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np

import multiprocessing as mp
#import tifffile as tf 
#Sfrom matplotlib import pyplot as plt
import h5py
from scipy.ndimage import median_filter, gaussian_filter, shift
from skimage.external.tifffile import TiffFile
from datetime import datetime

from os import listdir
from os.path import isfile, join
#import warnings


def MC_fallback(frame, ref, imCenter, sig, eps, sFactor, baseline, med_size):
    # manipulate frame to match image
    mf = median_filter(frame, size=med_size)
    logimg = np.log1p(np.maximum((mf-baseline)*sFactor, eps))
    lp = gaussian_filter(logimg, sigma=sig)
    adjimg = logimg - lp
    adjimg = adjimg - adjimg.min()
    adjimg = (np.expm1(adjimg)/sFactor) + baseline
    
    # find shifts:
    cim = np.fft.fft2(adjimg)*ref
    cim = abs(np.fft.ifft2(cim))
    xcpeak = np.array(np.unravel_index(np.argmax(np.fft.fftshift(cim)), ref.shape))
    disps = imCenter - xcpeak
    
    return {'original':np.uint16(shift(frame, (disps[0], disps[1]))),'shifts':disps,'filtered':np.uint16(shift(adjimg, (disps[0], disps[1])))}

    
def mc_helper2(args):
    return MC_fallback(*args)

def MC_allSess(fileNames, saveDir, suffix, saveMinMaxDir, saveTsDir, pool = None):
    # take a list of file names, and motion correct them    
    print('Opening parallel pool and preparing variables:')
    mpool = False
    if pool == None:
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
        mpool = True
    
    # load first tiff to create the reference:
    print('start processing files for '+suffix+':')
    for f in range(0,len(fileNames)):
        with TiffFile(fileNames[f]) as tif: 
            print('loading file ' + str(f+1))
            a = tif.asarray()
            # first file create refernce
            if f == 0: 
                print('creating reference image and defining parameters:')
                maxI = np.zeros((len(fileNames),a[0].shape[0],a[0].shape[1]))
                minI = np.zeros((len(fileNames),a[0].shape[0],a[0].shape[1]))
                tf = []
                
                # define parameters:
                imshape = a.shape[1:]
                imcenter = np.array(imshape)/2
                eps = 7./3 - 4./3 -1 
                maxval = a.max()
                sFactor = 1./maxval
                baseline = a.min()
                sig = 35
                med_size = 5
             
                # create the reference image
                aref = median_filter(a[0], size=med_size)
                logimg = np.log1p(np.maximum((aref-baseline)*sFactor, eps))
                lp = gaussian_filter(logimg, sigma=sig)
                adjimg = logimg - lp
                adjimg = adjimg - adjimg.min()
                adjimg = (np.expm1(adjimg)/sFactor) + baseline
                ref = np.fft.fft2(adjimg).conjugate()
            
            print('processing file '+ str(f+1))
            imList = [[img, ref,imcenter,sig,eps,sFactor,baseline, med_size] for img in a]
            #corectedTif, shift = pool.map(mc_helper,imList)
            results = pool.map(mc_helper2,imList)
            
            print('saving files')
            outputs = [Image.fromarray(s['original']) for s in results]
            outputs[0].save(saveDir +'/' +'m_'+suffix +'_'+str(f+1).zfill(4)+'.tif',save_all = True,append_images=outputs[1:])
            
            outputs = [Image.fromarray(s['filtered']) for s in results]
            outputs[0].save(saveDir +'/' +'mf_'+suffix +'_'+str(f+1).zfill(4)+'.tif',save_all = True,append_images=outputs[1:])
            
            outputs = [s['filtered'] for s in results]
            maxI[f,:,:] = np.max(outputs, axis=0)
            minI[f,:,:] = np.min(outputs, axis=0)
            
            outputs = [s['shifts'] for s in results]
            outputs = np.asarray(outputs)
            
            print('Extracting time stemps')
            for l in range(0,len(tif)):
                k = str(tif[l].tags.image_description)
                m = k.find('Time_From_Start')
                milis = datetime.strptime(k[18+m:m+31],"%H:%M:%S.%f")
                tf.append(((milis.microsecond)/1000000)+(milis.second)+(milis.minute*60))

            
            hf = h5py.File(saveDir +'/' +'shifts.hdf5')
            hf.create_group('/'+suffix +'/'+str(f+1).zfill(4))
            hf.create_dataset('/'+suffix +'/'+str(f+1).zfill(4)+'/yshift', data=outputs[:,1])
            hf.create_dataset('/'+suffix +'/'+str(f+1).zfill(4)+'/xshift', data=outputs[:,0])
            hf.close()
            del results, outputs, a, imList

    print('Done Processing '+suffix+'. saveing max-min image')
    I = Image.fromarray(np.max(maxI, axis=0) - np.min(minI, axis=0))
    I.save(saveMinMaxDir+'/minmax_'+suffix+'.tif')
    print('saveing timestemps')
    hf = h5py.File(saveTsDir +'/' +'tiffTs.hdf5')
    hf.create_group('/'+suffix)
    hf.create_dataset('/'+suffix+'/tiffts', data=tf)
    hf.close()
    
    if mpool:
        pool.close()
