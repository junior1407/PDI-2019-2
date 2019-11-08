#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Quase
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calcMean(data):
    prob = calcProbabilityArray(data)
    mean = 0
    for i in range(256):
        mean = mean + i*prob[i]
    return mean


def getNeighbordhoodData(data,size, neigh_no):
    height,width = img.shape[:2] 
    many_by_line = width/size
    center_x = int(size//2 + size * (neigh_no // many_by_line))
    center_y = int(size//2 + size * (neigh_no % many_by_line))
    data2 = data[ center_x - size//2:center_x + 1+  size//2 , 
                 center_y - size//2:center_y + 1+  size//2]
    return data2

def getNeighborhoodNo(data,size, x, y):
    height,width = data.shape[:2] 
    many_by_line = width/size
    return int( (x//many_by_line)*3 + (y//many_by_line))    

def calcMeanNeighborhood(data,size, center_no): #Center_no starts at 1
    return calcMean(getNeighbordhoodData(data,size, center_no))

def calcSDNeighborhood(data,size, center_no):
    return calcSD(getNeighbordhoodData(data,size, center_no))

def calcProbabilityArray(data):
    height,width = img.shape[:2]
    unique, counts = np.unique(data, return_counts=True) #Pixel Value, Frequency of Pixel Value
    N = height * width # Number of Pixels
    counts = counts / N # Probability of each pixel value.
    dic  = {k:0 for k in range(256)}
    for i in range(len(counts)):
        dic[unique[i]] = counts[i]
    return dic
    
def calcNthMoment(data, n):
    mean = calcMean(data)
    prob = calcProbabilityArray(data)
    moment = 0
    for i in range(256):
        moment = moment + prob[i]*(i -mean)**n
    return moment

def calcVariance(data):
    return calcNthMoment(data,2)

def calcSD(data):
    return calcVariance(data)**(1/2)

def checkCondition(x,y, data, size):   
    
    neighborhoodNo =getNeighborhoodNo(data, size, x,y)
    neighbordhoodData = getNeighbordhoodData(data, size, neighborhoodNo)
    k0 = 0
    m_g = calcMean(data)
    m_xy = calcMean(neighbordhoodData)
    k1 = 0.1
    k2 = 0
    sd_g = calcSD(data)
    sd_xy = calcSD(neighbordhoodData)
    k3 = 0.1
    C = 22.8
    if (k0 * m_g <= m_xy and m_xy<=k1*m_g and k2*sd_g<=sd_xy and sd_xy<=k3*sd_g ):
        return C * data[x][y]
    return data[x][y]
    

N = 3 #Neighborhood size

imgfile = '../db/cameraman_original.png'
img= cv2.imread(imgfile, 0)
height,width = img.shape[:2]

height = (height//N)*N
width = (height//N)*N
img = cv2.resize(img,(height, width))   
img2 = np.zeros((height,width), np.uint8)
for i in range(height):
    for j in range(width):
        img2[i][j] = checkCondition(i,j,img,N)

        



