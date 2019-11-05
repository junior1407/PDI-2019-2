#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:13:45 2019

@author: tiago
"""

#%% Digital Image Processing Utils
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
folder = "db"

#%matplotlib inline

plt.rcParams['figure.figsize'] = (18, 9)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

print("NumPy version: \t\t\t", np.__version__)
print("OpenCV version: \t\t", cv2.__version__)
print("Current working directory: \t", os.getcwd())
print("Python interpreter path: \t", sys.executable)
print("Folder containing images: \t", folder)
        

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

