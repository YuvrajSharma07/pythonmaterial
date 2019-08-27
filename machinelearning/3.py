#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[9]:


#DataFrame renders a table using the pandas(pd)


# In[10]:


df = pd.DataFrame({
    'x':[12,20,28,18,29],
    'y':[39,36,30,52,54]
})


# In[14]:


plt.scatter(df.x,df.y, color="#000000")
plt.plot(df.x,df.y)
plt.show()


# In[13]:


#use centroids to form clusters
#random centroids are created first


# In[15]:


#assignment stage


# In[ ]:




