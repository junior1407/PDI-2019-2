# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:47:33 2019

@author: Valdir
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def getNeighborhood(data, x,y, N):
    height,width = data.shape[:2]
    x_min = x - N//2
    x_min = 0 if x_min < 0 else x_min
    
    x_max = x + N//2
    x_max = height-1 if x_max +1 == height else x_max
    
    y_min = y - N//2
    y_min = 0 if y_min < 0 else y_min
    
    y_max = y + N//2
    y_max = width-1 if y_max +1 == width else y_max
    
    return data[x_min:x_max+1, y_min:y_max+1]
                
def medianFilter(img, N):
    height,width = img.shape[:2]
    new = np.ones((height,width), dtype='uint8')
    for i in range(height):
        for j in range(width):
            new[i][j] = np.median(getNeighborhood(img, i, j, N))
    return new
    

imgfile = '../db/jenny.jpg'
img= cv2.imread(imgfile, 0)
blurred = cv2.blur(img, (10,10))
#blurred = medianFilter(img, 3)
#lap = cv2.Laplacian(blurred,cv2.CV_64F)
lap = cv2.Laplacian(blurred,cv2.CV_16S)
filtered = img - 0.7*lap
plt.figure(figsize=(10,10))
plt.subplot(311), plt.imshow(blurred, cmap='gray')
plt.subplot(312), plt.imshow(lap, cmap='gray')
plt.subplot(313), plt.imshow(filtered, cmap='gray')
plt.show()
#plt.imsave()
