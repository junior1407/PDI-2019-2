import cv2
import matplotlib.pyplot as plt
import numpy as np


def cropCenter(img, x_center, y_center):
    y, x, c = img.shape
    xoff = (x - x_center) // 2
    yoff = (y - y_center) // 2
    img2= img[yoff:-yoff,xoff:-xoff]
    return img2

def swap(a,b):
    return b,a

def flipImg(img):
    height = width = img.shape[0]
    blank_image = np.zeros((height,width,3), np.uint8)
    for i in range(img.shape[0]):  #x
        for j in range(img.shape[1]): #y
             blank_image[i][j] = img[i][width-1-j]

    return blank_image
        

imfile = '../db/lena.png'
img = cv2.imread(imfile)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
#img2= cropimread(im_rgb,200,200)
img_flipped = flipImg(img)
img_cropped = cropCenter(img, 200, 200)

plt.figure(figsize=(8,8))
plt.subplot(221)
plt.title("Original")
plt.imshow(img)

plt.subplot(222)
plt.title("Flipped")
plt.imshow(img_flipped)

plt.subplot(223)
plt.title("Original")
plt.imshow(img)

plt.subplot(224)
plt.title("Cropped")
plt.imshow(img_cropped)
#plt.imshow(img2)
#Crop fli

