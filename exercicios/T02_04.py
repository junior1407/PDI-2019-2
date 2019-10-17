#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Imports
import numpy as np
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


#Function
def generateGaussian(theta,a, b, c):  
    global X,Y
    Xr = np.cos(theta)*X + np.sin(theta)*Y
    Yr = -np.sin(theta)*X + np.cos(theta)*Y
    X=Xr;
    Y=Yr;
    x_y = a * np.exp(-(X-b)**2/(2*(c**2)))
    return x_y
    
    
    


# In[16]:


#Image constants
height = 100
width = 100
n = 100
x = np.linspace(-height//2, height//2, n)
y = np.linspace(-width//2, width//2,n)
X, Y = np.meshgrid(x,y)


# In[17]:


img = generateGaussian(np.pi/2, 10,0,20)
plt.imshow(x_y, cmap = "gray")
plt.show()


# In[ ]:





# In[ ]:




