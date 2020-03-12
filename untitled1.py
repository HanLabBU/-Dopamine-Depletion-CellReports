#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:30:47 2020

@author: dana_z
"""

setup="""
import numpy as np
rSize = (12207,87,8898)
def alocateMem(rSize):
    arr = np.empty(rSize)
    for n in range(0,rSize[2]):
        tempArr = np.random.rand(rSize[0],rSize[1])
        arr[:,:,n] = tempArr
    
    return arr
"""
def dontAlocateMem(rSize):
    for n in range(0,rSize[2]):
        tempArr = np.random.rand(rSize[0],rSize[1],1)
        if 'arr' in locals():
            arr = np.concatenate((arr,tempArr), axis = 2)
        else:
            arr = tempArr

    return arr
