#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Imports
import numpy as np
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Image Config
height = 100
width = 100
n = 100
x = np.linspace(-height//2, height//2, n)
y = np.linspace(-width//2, width//2,n)
X, Y = np.meshgrid(x,y)


# In[5]:



rot= np.pi/4; #Rotate 45 degrees clockwise 
Xr = np.cos(rot)*X + np.sin(rot)*Y
Yr = -np.sin(rot)*X + np.cos(rot)*Y

x_y = np.sin(Xr)
plt.imshow(x_y, cmap = 'gray')
plt.show()


# In[ ]:




