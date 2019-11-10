# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calcProbabilityArray(data):
    height,width = img.shape[:2]
    unique, counts = np.unique(data, return_counts=True) #Pixel Value, Frequency of Pixel Value
    N = height * width # Number of Pixels
    counts = counts / N # Probability of each pixel value.
    arr  = [0 for k in range(256)]
    for i in range(len(counts)):
        arr[unique[i]] = counts[i]
    return arr

def calcHistTable(data, prob):
    s_ = [0 for i in range(256)] # arr = (prob_k, s_k)
    for k in range(256):
        s_k = 0
        for j in range(k+1):
            s_k+= prob[j]
        s_k *= 255
        s_k = round(s_k)
        s_[k] = int(s_k)
    return s_

def replace(data, s_k):
    height,width = data.shape[:2]
    blank = np.zeros((height,width), np.uint8)
    for x in range(height):
        for y in range(width):
            curr = data[x][y]
            blank[x][y] = s_k[curr]
    return blank    
        

imgfile = '../db/lena.png'
img= cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE  )
height,width = img.shape[:2]
prob = calcProbabilityArray(img) 
s_k = calcHistTable(img, prob)
new = replace(img, s_k)

plt.figure()
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.subplot(222), plt.imshow(new, cmap='gray')
plt.subplot(223), plt.hist(img.ravel(),256, [0, 256])
plt.subplot(224), plt.hist(new.ravel(),256, [0, 256])

