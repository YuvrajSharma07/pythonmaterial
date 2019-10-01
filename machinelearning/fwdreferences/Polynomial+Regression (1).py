
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy
from sklearn import linear_model
import pandas as pd


# In[67]:

housing=pd.read_csv('E:/AI/Workshop_AI_CODE/data/Housing_Data.csv',index_col=0)


# In[68]:

housing.head()


# In[82]:

X=housing[['lotsize']]
Y=housing[['price']]


# In[83]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y)


# In[84]:

from sklearn.linear_model import LinearRegression
lin=LinearRegression()


# In[85]:

lin.fit(xtrain,ytrain)


# In[86]:

lin.intercept_


# In[87]:

lin.coef_


# In[28]:

pred=lin.predict(xtest)


# In[34]:

lin.predict([[30000,5,2,2]])


# In[29]:

from sklearn import metrics
metrics.mean_absolute_error(pred,ytest)


# In[90]:

from sklearn.preprocessing import PolynomialFeatures

pol=PolynomialFeatures(degree=3)
X_pol=pol.fit_transform(X)


# In[91]:

pol.fit(X_pol,Y)


# In[92]:

linp=LinearRegression()
linp.fit(X_pol,Y)


# In[93]:

Y.shape


# In[96]:

plt.scatter(X,Y)
plt.plot(X,linp.predict(X_pol),color='green')
plt.show()


# In[18]:

import numpy as np
X=np.array([[3,5,6.6,7,8.5]]).T
Y=np.array([[2,4.3,6.7,8,9.8]]).T


# In[19]:

from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression  
pol = PolynomialFeatures(degree = 5) 
X_pol = pol.fit_transform(X) 
  
pol.fit(X_pol, Y) 
linp = LinearRegression() 
linp.fit(X_pol, Y) 


# In[20]:

X_pol


# In[21]:

plt.scatter(X, Y) 
plt.plot(X, linp.predict(X_pol), color = 'red') 
plt.show()


# In[ ]:




# In[ ]:



