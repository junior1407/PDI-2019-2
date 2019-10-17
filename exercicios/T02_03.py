#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Imports
import numpy as np
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Image configs
height = 100
width = 100
n = 100
x = np.linspace(-height//2, height//2, n)
y = np.linspace(-width//2, width//2,n)
X, Y = np.meshgrid(x,y)


# In[14]:


#Gaussian curve arguments
a = 10
b = 0
c = 20
x_y = a * np.exp(-(X-b)**2/(2*(c**2)))

plt.imshow(x_y, cmap = "gray")
plt.show()


# In[ ]:





# In[ ]:




