# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Batch compression of Phallusia images for Supplementary Figure 5        %
% for B³D paper                                                           %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 28.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""



import h5py
import os

#%% parameters

comprLevels = (0.0, 0.5, 1,0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
#comprLevels = (4.0, 5.0)
#comprLevels = (0.5, 1.0)
stacks = ("S0", "S1")
cams = ("CL", "CR")

tileSize = 24
chunkSize = (1,1600,1800)


compressionType = 101
quantStep = 1    # q=1*sigma
conversion = 2.1845  # 65535/30000
bgLevel = 99
readNoise = 1.6 # in e-
tileSize = 64

#%% compress
for stack in stacks:
    for cam in cams:
        filename =  "D:/GPU_compression/!paper_figures/SFig_5_Phallusia/%s%s/phallusia_T05_%s%s_00005.h5" % (stack, cam, stack, cam)
        fOrig = h5py.File(filename, "r")
        dset = fOrig['Data']
        
        for comprLevel in comprLevels:
            quantStep = comprLevel * 1000            
            
            directory = "D:/GPU_compression/!paper_figures/SFig_5_Phallusia/compressed/%d/%s%s" % (quantStep, stack, cam)
            # check if directory exists, if not create it
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            # check if output file exists, if yes, delete
            newFile = "%s/phallusia_T05_%s%s_00005.h5" % (directory, stack, cam)
            if os.path.isfile(newFile):
                os.remove(newFile)
                
            
            # create file
            f = h5py.File(newFile)
            
            # lossless compression
            
                              
            
            dsetNew = f.create_dataset("Data", data=dset.value, chunks=chunkSize,
                                    compression=333,
                                    compression_opts=(0, compressionType, quantStep, round(conversion*1000), bgLevel, round(readNoise*1000), tileSize))
            
           
            #%
            f.close()
            
        fOrig.close()
        
#%% decompress images

saveFolder = "D:/GPU_compression/!paper_figures/SFig_5_Phallusia/decompressed"
loadFolder = "D:/GPU_compression/!paper_figures/SFig_5_Phallusia/compressed"

for stack in stacks:
    for cam in cams:
        for comprLevel in comprLevels:
            quantStep = comprLevel * 1000
            filename =  "%s/%d/%s%s/phallusia_T05_%s%s_00005.h5" % (loadFolder, quantStep, stack, cam, stack, cam)
            fOrig = h5py.File(filename, "r")
            dset = fOrig['Data']    
            
            directory = "%s/%d/%s%s" % (saveFolder, quantStep, stack, cam)
            # check if directory exists, if not create it
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            # check if output file exists, if yes, delete
            newFile = "%s/phallusia_T05_%s%s_00005.h5" % (directory, stack, cam)
            if os.path.isfile(newFile):
                os.remove(newFile)
            
            # create file
            f = h5py.File(newFile)
            dsetNew = f.create_dataset("Data", data=dset.value, chunks=chunkSize)
           
            #%
            f.close()
            
        fOrig.close()