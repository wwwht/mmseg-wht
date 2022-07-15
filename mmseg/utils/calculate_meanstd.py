# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:38:17 2021

@author: 17478build
"""
import os
import cv2
import numpy as np
from tqdm import tqdm  # pip install tqdm
import argparse
import ipdb

# def input_args():
#     parser = argparse.ArgumentParser(description="calculating mean and std")
#     parser.add_argument("--data_fmt",type=str,default='samples_{name}')
#     parser.add_argument("--data-name",type=str,nargs="+",default=['morning','noon','afternoon','dusk','snowy'])
#     return parser.parse_args()

DATA_DIR = os.path.abspath('A:/datasets/hubmapOrganSeg/mmdata/images')
DATA_NAME = ['images']

if __name__ == "__main__":
    # opt = input_args()
    img_files =[]
    # for name in opt.data_name:
    # for name in DATA_NAME:
    img_dir = DATA_DIR
    # img_dir = opt.data_fmt.format(name=name)
    print(img_dir)
    files = os.listdir(img_dir)
    img_files.extend([os.path.join(img_dir,file) for file in files])

    meanRGB = np.asarray([0,0,0],dtype=np.float64)
    varRGB = np.asarray([0,0,0],dtype=np.float64)
    for img_file in tqdm(img_files,desc="calculating mean",mininterval=0.1):
        img = cv2.imread(img_file,-1)
        meanRGB[0] += np.mean(img[:,:,0])/255.0
        meanRGB[1] += np.mean(img[:,:,1])/255.0
        meanRGB[2] += np.mean(img[:,:,2])/255.0
    meanRGB = meanRGB/len(img_files)
    for img_file in tqdm(img_files,desc="calculating var",mininterval=0.1):
        img = cv2.imread(img_file,-1)
        varRGB[0] += np.sqrt(np.mean((img[:,:,0]/255.0-meanRGB[0])**2))
        varRGB[1] += np.sqrt(np.mean((img[:,:,1]/255.0-meanRGB[1])**2))
        varRGB[2] += np.sqrt(np.mean((img[:,:,2]/255.0-meanRGB[2])**2))
    varRGB = varRGB/len(img_files)
    print("meanRGB:{}".format(meanRGB))
    print("stdRGB:{}".format(varRGB))
