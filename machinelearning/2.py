#!/usr/bin/env python
# coding: utf-8

# In[22]:


#Supervised Leaning
#Regression
#Linear reg


# In[23]:


#Numpy - to process multi dimensional arrays/data types/matrices
#pandas - to access datas as data frame
#Matplotlib - to plot over X and Y axis


# In[96]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[97]:


#Feature
x = np.array([11, 13, 45, 23, 32])
#Label
y = np.array([12, 17, 34, 26, 27])


# In[98]:


x_mean = x.mean()
y_mean = y.mean()


# In[99]:


num = 0
den = 0
for i in range(len(x)):
    num += (x[i]-x_mean)*(y[i]-y_mean)
    den += (x[i]-x_mean)*(x[i]-x_mean)

m = num/den
c = y_mean - (m*x_mean)

#m = slope
#c = intercept


# In[100]:


X = np.linspace(x.min()-10, x.max()+10)
Y = m*X + c


# In[101]:


plt.plot(X,Y, color="#fd4207")
plt.scatter(x,y)
plt.show()


# In[102]:


a = np.array([[34, 56, 7, 8, 89, 90], [31, 5, 72, 81, 9, 50], [21, 51, 2, 41, 19, 5]])


# In[103]:


#rendering a given data file, this is a csv file, use read_table for tables, and so on

data = pd.read_csv('Advertising.csv')
data.head()


# In[104]:


from sklearn.model_selection import train_test_split
tvtrain,tvtest,satrain,satest = train_test_split(tv,sa,test_size=0.2)


# In[105]:


data = data.drop('Unnamed: 0', axis=1)
tv = data[['TV']]
sa = data[['sales']]
ne = data[['newspaper']]
tv_mean = tv.mean()
ne_mean = ne.mean()
sa_mean = sa.mean()
#drop() is to drop a data set


# In[106]:


plt.scatter(tv, sa, color="#008080")
plt.scatter(ne, sa, color="#333333")
plt.show()


# In[107]:


lin = LinearRegression()
#Training
lin.fit(tv,sa)


# In[108]:


print(lin.intercept_)
print(lin.coef_)


# In[109]:


lin.predict([[50]])


# In[110]:


tvtrain.shape


# In[111]:


sapred = lin.predict(tvtest)

#sapred is the predicted value
#satest is the real value


# In[112]:


from sklearn.preprocessing import PolynomialFeatures


# In[113]:


pol = PolynomialFeatures(degree = 4)


# In[114]:


tv_pol = pol.fit_transform(tv)


# In[115]:


iris = pd.read_csv('Iris.csv')


# In[116]:


iris.drop('Id', axis=1, inplace=True)
iris.head()


# In[117]:


#iris.dtypes gives data types of the given set


# In[118]:


#mapping

iris['Mapped'] = iris.Species.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
# the iris['Mapped'] adds another col


# In[119]:


ix = iris.drop(['Species', 'Mapped'], axis=1)
iy = iris[['Mapped']]


# In[120]:


ixtrain,ixtest,iytrain,iytest = train_test_split(ix, iy, test_size=0.2)


# In[121]:


#var.shape gives the dimensions of the variable/object


# In[122]:


#SVM - Scale Vector Machine


# In[123]:


#knearest method - finding output of nearest elements of the target to get the output for the same


# In[124]:


#forest method - where the vote of each tree is taken out and output is decided 


# In[125]:


from sklearn.neighbors import KNeighborsClassifier


# In[126]:


knn = KNeighborsClassifier(n_neighbors = 3)


# In[127]:


knn.fit(ixtrain, iytrain)


# In[128]:


knn.predict([[2,3,2.4,1.2]])


# In[129]:


iypred = knn.predict(ixtest)


# In[130]:


iytest = iytest.values


# In[131]:


count = 0
for i in range(30):
    if iytest[i] == iypred[i]:
        count += 1
count/len(ixtest)


# In[132]:


knn.score(ixtest, iytest)


# In[133]:


acc = []


# In[134]:


for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(ixtrain, iytrain)
    acc.append(knn.score(ixtest, iytest))
numb = np.arange(1,15)
plt.plot(numb, acc)
plt.show()


# In[85]:


k = np.array(['setosa', 'versicolor', 'virginica'])


# In[86]:


k[0]
#add the array of for predict


# In[ ]:





# In[ ]:





# In[ ]:




