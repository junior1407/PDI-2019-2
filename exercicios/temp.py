#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Quase
import cv2
import matplotlib.pyplot as plt
import numpy as np



def calcMean(data, prob):
    #prob = calcProbabilityArray(data)
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

def getwindowNo(data,size, x, y):
    
    height,width = data.shape[:2] 
    #many_by_line = width/size
    #return int( (x//many_by_line)*many_by_line + (y//size))    
    col = y//size
    line = x//size
    cabe_numa_linha = width //size
    qnts_antes = line * cabe_numa_linha
    
    #print(col,line, cabe_numa_linha, qnts_antes)
    return qnts_antes + line

#def calcMeanNeighborhood(data,size, center_no): #Center_no starts at 1
#    return calcMean(getNeighbordhoodData(data,size, center_no))

#def calcSDNeighborhood(data,size, center_no):
#    return calcSD(getNeighbordhoodData(data,size, center_no))

def calcProbabilityArray(data):
    height,width = img.shape[:2]
    unique, counts = np.unique(data, return_counts=True) #Pixel Value, Frequency of Pixel Value
    N = height * width # Number of Pixels
    counts = counts / N # Probability of each pixel value.
    dic  = {k:0 for k in range(256)}
    for i in range(len(counts)):
        dic[unique[i]] = counts[i]
    return dic
    
def calcNthMoment(data, n, mean, prob):
    moment = 0
    for i in range(256):
        moment = moment + prob[i]*(i -mean)**n
    return moment

def calcVariance(data, mean, prob):
    return calcNthMoment(data,2, mean, prob)

def calcSD(data,  mean, prob):
    return calcVariance(data, mean, prob)**(1/2)

def checkCondition(x,y, data, size):   
    
    global dataLib
    global meanLib
    global probLib
    global sdLib
    global k0,p_g,m_g ,k1,k2,sd_g,k3,C

    windowNo =getwindowNo(data, size, x,y)
    if (windowNo not in dataLib):
        dataLib[windowNo] = getNeighbordhoodData(data, size, windowNo)
    if (windowNo not in probLib):
        probLib[windowNo] = calcProbabilityArray(dataLib[windowNo])
    if (windowNo not in meanLib):
        meanLib[windowNo] = calcMean(dataLib[windowNo], probLib[windowNo])
    if (windowNo not in sdLib):
        sdLib[windowNo] = calcSD(data, meanLib[windowNo], probLib[windowNo])
    
    if (k0 * m_g <= meanLib[windowNo] and meanLib[windowNo]<=k1*m_g and k2*sd_g<=sdLib[windowNo] and sdLib[windowNo]<=k3*sd_g ):
        return C * data[x][y]
    return data[x][y]
    

N = 5 #Neighborhood size

imgfile = '../db/squares_noisy.tif'
img= cv2.imread(imgfile, 0)
height,width = img.shape[:2]

height = (height//N)*N
width = (height//N)*N
img = cv2.resize(img,(height, width))   
img2 = np.zeros((height,width), np.uint8)

meanLib = {}
sdLib = {}
dataLib = {}
probLib = {}
k0 = 0
p_g = calcProbabilityArray(img)
m_g = calcMean(img, p_g)
k1 = 0.1
k2 = 0
sd_g = calcSD(img, m_g, p_g)
k3 = 0.1
C = 22.8

for i in range(height):
    for j in range(width):
        img2[i][j] = checkCondition(i,j,img,N)

plt.figure(figsize=(20,10))
plt.subplot(121), plt.title("Imagem original"), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.title("Hist Loc Equalizado"), plt.imshow(img2, cmap='gray')

plt.show()
        



