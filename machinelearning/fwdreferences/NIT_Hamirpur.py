
# coding: utf-8

# In[2]:

a=30
b=6
print(a+b)
print(a-b)
print(a/b)


# In[13]:

#Pandas
import pandas as pd
ipl = pd.read_csv(r'E:\ML_Codes\ipl\matches.csv')


# In[9]:

ipl.head()


# In[10]:

len(ipl)


# In[11]:

ipl.shape


# In[16]:

ipl.season.nunique()


# In[19]:

ipl.season.value_counts().idxmax()


# In[27]:

ipl.iloc[ipl.win_by_runs.idxmax()].winner


# In[25]:

ipl.iloc[43].winner


# In[35]:

ipl.iloc[ipl[ipl.win_by_runs > 0].win_by_runs.idxmin()].winner


# In[38]:

ipl[ipl.win_by_runs == 1].shape


# In[42]:

ipl.winner.value_counts().idxmax()


# In[47]:

len(ipl[ipl.toss_winner == ipl.winner]) / len(ipl)


# In[48]:

data=pd.DataFrame({'x':[3,4.3,5,6,7.8,8],
                  'y':[3.6,5,6.2,7.4,8.3,8.9]})


# In[50]:

import matplotlib.pyplot as plt
plt.scatter(data.x, data.y)
plt.show()


# In[52]:

X_mean = data.x.mean() 
Y_mean = data.y.mean()


# In[57]:

num = 0
den = 0
for i in range(len(data)):
    num += (data.x[i]-X_mean)*(data.y[i]-Y_mean)
    den += (data.x[i]-X_mean)*(data.x[i]-X_mean)
m=num/den
print(m)
c = Y_mean - m * X_mean
print(c)


# In[60]:

import numpy as np
X = np.linspace(data.x.min(),data.x.max())
Y=m*X+c
plt.plot(X,Y,color='r')
plt.scatter(data.x,data.y)
plt.show()


# In[59]:

X


# In[63]:

data=pd.read_csv(r'E:\ML_Codes\advertising\Advertising.csv',
                index_col=0)


# In[67]:

data.head()


# In[106]:

X=data[['TV','radio','newspaper']]
Y=data[['sales']]


# In[107]:

Y.shape


# In[108]:

from sklearn.linear_model import LinearRegression
lin = LinearRegression()


# In[109]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)


# In[110]:

lin.fit(xtrain,ytrain)


# In[111]:

lin.intercept_


# In[112]:

lin.coef_


# In[113]:

#Mean Absolute error  -  MAE
#Mean Squared Error  -  MSE
#Root Mean Squared Error  - RMSE
ypred=lin.predict(xtest)


# In[114]:

from sklearn import metrics
metrics.mean_absolute_error(ypred,ytest)


# In[105]:

lin.predict([[4.5,3],[6,4.5],[7.5,8]])


# In[116]:

plt.scatter(data.TV,data.sales)
plt.show()


# In[ ]:



