#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:44:07 2019

@author: junior1407
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def convertBGRtoRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def convertRGBtoBGR(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def plotHistGrey(img):
    #histr = cv2.calcHist([img], [0], None, [256], [0,256])
    #plt.plot(histr)
    #plt.xlim([0, 256])
    #plt.show()
    #alternate way
    plt.hist(img.ravel(),256, [0, 256])
    plt.show()

def plotHistColor(img):
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    
    
def cropCenter(img, x_center, y_center):
    y, x, c = img.shape
    xoff = (x - x_center) // 2
    yoff = (y - y_center) // 2
    img2= img[yoff:-yoff,xoff:-xoff]
    return img2

def crop(img, x_start, x_end, y_start, y_end):
    img2= img[y_start:y_end,x_start:x_end]
    return img2

def flipImg(img):
    height = width = img.shape[0]
    blank_image = np.zeros((height,width,3), np.uint8)
    for i in range(img.shape[0]): 
        for j in range(img.shape[1]):
             blank_image[i][j] = img[i][width-1-j]

    return blank_image
    
def translate(img, Tx, Ty):
    height,width = img.shape[:2]
    T = np.float32([[1, 0 , Tx],[0, 1, Ty]])
    return cv2.warpAffine(img, T, (width, height))

def rotate(img, degrees):
    height,width = img.shape[:2]
    T=cv2.getRotationMatrix2D((width/2, height/2),degrees,1)
    return cv2.warpAffine(img, T, (width, height))

def resizeByScale(img, scale):
    return cv2.resize(img, None, fx = scale, fy = scale)

def resize(img, width, height):
    return cv2.resize(img,(width, height))

def applyMask(img, mask):
    height,width = img.shape[:2]
    blank_image = np.zeros((height,width,3), np.uint8)
    for i in range(height):
        for j in range(width):
            if (mask[i][j]):
                blank_image[i][j] = img[i][j]
    return blank_image

def getNthSlice(data, n):
    # gets only the nth bit
    return np.bitwise_and(data, 1 << n)

def clearNthSlice(data, n):
    # Clear nth Bit
    return np.bitwise_and(data, ~(1<<n))

def plotAllSlices(data):
    # Plots all bits as separate images
    plt.figure(figsize=(15,15))
    for i in range(0, 7+1):        
        plt.subplot(4,4,i+1), plt.title(str(i)+"th bit"), plt.imshow(getNthSlice(data, i), cmap= "gray")
    plt.show()
    
#Getsthe probability array
def _calcProbabilityArray(data):
    height,width = data.shape[:2]
    unique, counts = np.unique(data, return_counts=True) #Pixel Value, Frequency of Pixel Value
    N = height * width # Number of Pixels
    counts = counts / N # Probability of each pixel value.
    arr  = [0 for k in range(256)]
    for i in range(len(counts)):
        arr[unique[i]] = counts[i]
    return arr

#Computes the hist table:  s_k
def _calcHistTable(data):
    prob = _calcProbabilityArray(data)
    s_ = [0 for i in range(256)] # arr = (prob_k, s_k)
    for k in range(256):
        s_k = 0
        for j in range(k+1):
            s_k+= prob[j]
        s_k *= 255
        s_k = round(s_k)
        s_[k] = int(s_k)
    return s_


#Maps values from sk_1 to their closes in sk_2
def _getMapping(sk_1, sk_2):
    mapp = [0 for i in range(256)]
    for i in range(256):
        s_original = sk_1[i]
        best_s_index = 0
        best_s_value = sk_2[0]
        for j in range(256):
            s_cmp = sk_2[j]
            #Saving the closest
            if (abs(s_cmp - s_original) <
                abs(best_s_value - s_original)):
                best_s_index = j
                best_s_value = s_cmp
            #When their diff = 0, The highest is chosen 
            elif ((abs(s_cmp - s_original) ==
                abs(best_s_value - s_original)) and 
                s_cmp > best_s_value):
                best_s_index = j
                best_s_value = s_cmp
        mapp[i] = best_s_index
    return mapp

#Replace all pixel values with their values from the hist_table
def _replaceMatch(data, s_k ):
    height,width = data.shape[:2]
    blank = np.zeros((height,width), np.uint8)
    for x in range(height):
        for y in range(width):
            curr = data[x][y]
            blank[x][y] = s_k[curr]
    return blank    




def getGradientX(img, ddepth= cv2.CV_16S, K=3, S=1, D=0):
    return cv2.Sobel(img, ddepth, 1,0, ksize=K,scale =S, delta=D, borderType=cv2.BORDER_DEFAULT)

def getGradientY(img, ddepth=cv2.CV_16S, K=3, S=1, D=0):
    return cv2.Sobel(img, ddepth, 0,1, ksize=K,scale =S, delta=D, borderType=cv2.BORDER_DEFAULT)

def sobelFilter(img,k_x=0.5, k_y=0.5):
    grad_x = getGradientX(img)
    grad_y = getGradientY(img)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, k_x, abs_grad_y, k_y, 0)
   
def plotContours(img):
    grad = sobelFilter(img)
    plt.title('Contours'), plt.imshow(grad, cmap='gray'), plt.show()

def histogramMatch(img1, img2):
    height,width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height ))
    s_k_1 = _calcHistTable(img1)
    s_k_2 = _calcHistTable(img2)
    mapping = _getMapping(s_k_1,s_k_2)
    new =  _replaceMatch(img1, mapping)
    return new

def laplacianSharpenning(img, dtype = cv2.CV_16S, intensity = 0.7):
    lap = cv2.Laplacian(img,dtype)
    return img - intensity* lap

#Replace all pixel values with their values from the hist_table
def globalEqualization(data):
    s_k = _calcHistTable(data)
    height,width = data.shape[:2]
    blank = np.zeros((height,width), np.uint8)
    for x in range(height):
        for y in range(width):
            curr = data[x][y]
            blank[x][y] = s_k[curr]
    return blank    

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

def averageFilter(data, N):
    kernel = np.ones((N,N), np.float32)/(N**2)
    return cv2.filter2D(data, cv2.CV_32F, kernel)

def medianFilter(data, N):
    return cv2.medianBlur(data, N)
    


# Improve Contranst  lady.png
# Filter to Sharpen eye.png
# Bit Slice
# Gradients (separate) and Gradients together(abs(x)+abs(y))

# 
