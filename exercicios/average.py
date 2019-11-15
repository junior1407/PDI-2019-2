# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:47:33 2019

@author: Valdir
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def addSaltnPepper(data, saltiness, spiceness):
    height,width = data.shape[:2]
    noise = np.zeros((height,width), data.dtype)
    cv2.randu(noise,0,255)        
    salt = noise > saltiness
    pepper = noise < spiceness
    
    img2 = data.copy()
    img2[salt] = 255
    img2[pepper] = 0
    return img2

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
                
def averageFilter(img, N):
    height,width = img.shape[:2]
    new = np.ones((height,width), dtype='uint8')
    for i in range(height):
        for j in range(width):
            new[i][j] = np.mean(getNeighborhood(img, i, j, N))
    return new
    

imgfile = '../db/jenny.jpg'
img= cv2.imread(imgfile, 0)
noisy = addSaltnPepper(img, 240,15)

N = 5# Neighborhood size NXN
filtered = averageFilter(noisy, N)
plt.figure()
plt.subplot(121), plt.imshow(noisy, cmap='gray')
plt.subplot(122), plt.imshow(filtered, cmap='gray')
plt.show()
