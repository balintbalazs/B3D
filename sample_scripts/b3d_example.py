# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:37:34 2017

@author: Balint Balazs
"""
import os
import h5py

#%% load uncompressed dataset
baseFolder = "../"
inFile = "B3D_demo_drosophila_uncompressed.h5"
f = h5py.File(baseFolder+inFile, "r+")
dset = f['Data']

# set chunk size to the same as input
chunks = dset.chunks

#%% This will remove outifle if it already exists
outFile = "B3D_demo_drosophila_compressed.h5"
if (os.access(baseFolder+outFile, os.F_OK)):
    os.remove(baseFolder+outFile)
    
#%% lossless compression
f2 = h5py.File(baseFolder+outFile)
dset2= f2.create_dataset("B3D_Lossless", data=dset.value, chunks=chunks,
                        compression=32016)
f2.close()
#%% within noise level compression, Mode 1
compressionMode = 1
quantStep = 1    # q=1*sigma
conversion = 2.1845  # in DN/e-, here 65535/30000
bgLevel = 0 # set it to camera average background level
readNoise = 1.6 # in e-
outPath = baseFolder+outFile

f2 = h5py.File(outPath)
dset2= f2.create_dataset("B3D_Mode1_1.00", data=dset.value, chunks=chunks,
                        compression=32016,
                        compression_opts=(round(quantStep*1000), compressionMode, round(conversion*1000), bgLevel, round(readNoise*1000)))
f2.close()
#%% within noise level compression, Mode 2
compressionMode = 2
quantStep = 1    # q=1*sigma
conversion = 2.1845  # 65535/30000
bgLevel = 0 # set it to camera average background level
readNoise = 1.6 # in e-
outPath = baseFolder+outFile

f2 = h5py.File(outPath)
dset2= f2.create_dataset("B3D_Mode2_1.00", data=dset.value, chunks=chunks,
                        compression=32016,
                        compression_opts=(round(quantStep*1000), compressionMode, round(conversion*1000), bgLevel, round(readNoise*1000)))
f2.close()
#%%
f.close()