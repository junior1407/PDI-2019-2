#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

imgfile = '../db/cameraman_original.png'
secretMessage = '../db/gt.tif'

img_original = cv2.imread(imgfile, 0)
height,width = img_original.shape[:2]
        

img_secret = cv2.imread(secretMessage, 0)
img_secret = cv2.resize(img_secret, (width, height))
img_secret = img_secret > 128
#img_secret = img_secret.astype("uint8")

for i in range(height):
    for j in range(width):
        if (img_secret[i][j]==True):    
            img_original |= 1<<0
        else:
            img_original[i][j] &= ~(1<<0)


plt.subplot(121), plt.imshow(img_original, cmap="gray")
plt.subplot(122), plt.imshow(img_secret, cmap="gray")

plt.show()
#
#cv2.resize(img,(width, height))