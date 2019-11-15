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


imgfile = '../db/lena.png'
img= cv2.imread(imgfile, 0)
new= addSaltnPepper(img, 220,10)
plt.imshow(img2, cmap='gray')
plt.show()


