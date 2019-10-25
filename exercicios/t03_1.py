import cv2
import matplotlib.pyplot as plt
import numpy as np


def cropimread(img, xcrop, ycrop):
    "Function to crop center of an image file"
    ysize, xsize, chan = img.shape
    xoff = (xsize - xcrop) // 2
    yoff = (ysize - ycrop) // 2
    img2= img[yoff:-yoff,xoff:-xoff]
    return img2

imfile = '../db/lena.png'
img = cv2.imread(imfile)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2= cropimread(im_rgb,200,200)
plt.imshow(img2)
#Crop and Flip

