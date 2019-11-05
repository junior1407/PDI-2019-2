#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:14:51 2019
@author: tiago
"""

#%% Lecture 001
runfile('dip.py')

filename = os.path.join(folder, "baboon.png")

#%% Read a grayscale image
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if type(img) is not np.ndarray:
    print("Error: Image not found!")

plt.imshow(img)
plt.show()
# cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
# cv2.imshow("img", img)
# #cv2.waitKey(0)
# while True:
#     if 0xFF & cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()

#%% Read a color image
img = cv2.imread(filename, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#%%
bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

#%% Splitting RGB channels
bgr = cv2.imread(filename, cv2.IMREAD_COLOR)

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb_lst = cv2.split(rgb)

plt.figure(1, figsize=(18,9)), plt.clf
plt.subplot(232), plt.imshow(rgb), plt.title('RGB')
plt.subplot(234), plt.imshow(rgb_lst[0], cmap='gray'), plt.title('R')
plt.subplot(235), plt.imshow(rgb_lst[1], cmap='gray'), plt.title('G')
plt.subplot(236), plt.imshow(rgb_lst[2], cmap='gray'), plt.title('B')
plt.show()

#%% Using a function handle
bgr2rgb = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
plt.figure(2)
plt.imshow(bgr2rgb(bgr))
plt.show()

#%% Splitting RGB channels
bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(rgb)

plt.figure(1), plt.clf
plt.subplot(232), plt.imshow(rgb), plt.title('RGB')
plt.subplot(234), plt.imshow(r, cmap='gray'), plt.title('R')
plt.subplot(235), plt.imshow(g, cmap='gray'), plt.title('G')
plt.subplot(236), plt.imshow(b, cmap='gray'), plt.title('B')
plt.show()
