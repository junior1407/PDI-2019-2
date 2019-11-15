# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:47:33 2019

@author: Valdir
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


imgfile = '../db/lena.png'
img= cv2.imread(imgfile, 0)
K=3 #Kernel Size = 3
img = cv2.GaussianBlur(img, (K,K), cv2.BORDER_CONSTANT)


ddepth = cv2.CV_16S #16 bits com sinal.
grad_x = cv2.Sobel(img, ddepth, 1,0, ksize=3,scale =1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(img, ddepth, 0,1, ksize=3,scale =1, delta=0, borderType=cv2.BORDER_DEFAULT)
    
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)


grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
plt.imshow(grad, cmap='gray')
plt.show()


