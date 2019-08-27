#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Supervised Leaning
#Regression
#Linear reg


# In[16]:


#Numpy - to process multi dimensional arrays/data types/matrices
#pandas - to access datas as data frame
#Matplotlib - to plot over X and Y axis


# In[44]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[45]:


#Feature
x = np.array([11, 13, 45, 23, 32])
#Label
y = np.array([12, 17, 34, 26, 27])


# In[46]:


x_mean = x.mean()
y_mean = y.mean()


# In[47]:


num = 0
den = 0
for i in range(len(x)):
    num += (x[i]-x_mean)*(y[i]-y_mean)
    den += (x[i]-x_mean)*(x[i]-x_mean)

m = num/den
c = y_mean - (m*x_mean)

#m = slope
#c = intercept


# In[48]:


X = np.linspace(x.min()-10, x.max()+10)
Y = m*X + c


# In[49]:


plt.plot(X,Y, color="#fd4207")
plt.scatter(x,y)
plt.show()


# In[50]:


a = np.array([[34, 56, 7, 8, 89, 90], [31, 5, 72, 81, 9, 50], [21, 51, 2, 41, 19, 5]])


# In[51]:


#rendering a given data file, this is a csv file, use read_table for tables, and so on

data = pd.read_csv('Advertising.csv')
data.head()


# In[52]:


from sklearn.model_selection import train_test_split
tvtrain,tvtest,satrain,satest = train_test_split(tv,sa,test_size=0.2)


# In[53]:


data = data.drop('Unnamed: 0', axis=1)
tv = data[['TV']]
sa = data[['sales']]
ne = data[['newspaper']]
tv_mean = tv.mean()
ne_mean = ne.mean()
sa_mean = sa.mean()
#drop() is to drop a data set


# In[64]:


plt.scatter(tv, sa, color="#008080")
plt.scatter(ne, sa, color="#333333")
plt.show()


# In[65]:


lin = LinearRegression()
#Training
lin.fit(tv,sa)


# In[66]:


print(lin.intercept_)
print(lin.coef_)


# In[67]:


lin.predict([[50]])


# In[68]:


tvtrain.shape


# In[69]:


sapred = lin.predict(tvtest)

#sapred is the predicted value
#satest is the real value


# In[74]:


plt.scatter(sapred, satest)
plt.show()


# In[ ]:





# In[ ]:




