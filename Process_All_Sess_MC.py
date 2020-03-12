#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:38:36 2018

@author: dana_z
"""

import numpy as np

import multiprocessing as mp
import tifffile as tf 
from matplotlib import pyplot as plt
import h5py
from scipy.ndimage import median_filter, gaussian_filter, shift
from PiplineScripts.pipelines import *
from os import listdir
from os.path import isfile, join, isdir
import re
import sqlalchemy as db
import sqlite3

user = 'auto_processing'
password = 'dz_preProcess'

engine = db.create_engine('mysql+pymysql://'+user+':'+password+'@localhost/preProcess')


# define aids and parameters:
mypath = '/home/dana_z/handata2/Dana/'
p = re.compile('\d{4,}$') #only mice folders...
s = re.compile('(_day|_Baseline)') #only 60HDA related sessions
miceNames = [f for f in listdir(mypath) if isdir(join(mypath, f)) and p.match(f)]
#print(miceNames)

conn = engine.connect()
meta = db.MetaData(engine, reflect =True)
table = meta.tables['mice']
dtable = meta.tables['data']
dMice = db.select([table.c.mouse_num])
res = conn.execute(dMice)
dmice = [str(_row[0]).zfill(4) for _row in res]

dSess = db.select([dtable.c.Suffix]).where(dtable.c.MC == 1)
res = conn.execute(dSess)
dsess = [str(_row[0]).zfill(4) for _row in res]



n_cores = 20
pool = mp.Pool(processes=n_cores)
for m in miceNames:
    if not m in dmice:
        fmice =table.insert().values(mouse_num=int(m))
        conn.execute(fmice)
    spath = join(mypath,m)
    sessNames = [f for f in listdir(spath) if isdir(join(spath, f)) and s.search(f)]
    for sess in sessNames:
        sesspath = join(spath,sess)
        fileNames = [join(sesspath, f) for f in listdir(sesspath) if isfile(join(sesspath, f)) and f[-4:] == '.tif' and f.find('green')==-1]
        suffix = m + sess[s.search(sess).span()[0]:]
        
        if suffix in dsess or len(fileNames)==0:
            continue
        
        print('start motion correction on: ',suffix)
        MC_allSess(fileNames, '/home/dana_z/HD1/Processed_tifs', suffix, '/home/dana_z/HD1/min_max', '/home/dana_z/HD1', pool)
        isess = dtable.insert().values(mouse_num=int(m),
                                       session =sess[s.search(sess).span()[0]+1:],
                                       Suffix = suffix,
                                       MC = 1)
        conn.execute(isess)
pool.close()
        

    
    