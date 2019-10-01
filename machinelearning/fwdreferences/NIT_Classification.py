
# coding: utf-8

# In[3]:

import pandas as pd
iris = pd.read_csv(r'E:\ML_Codes\iris-species\Iris.csv',
                  index_col = 0)


# In[8]:

iris['mapped']=iris.Species.map({'Iris-setosa' : 0, 
                                 'Iris-versicolor': 1,
                                 'Iris-virginica' : 2})


# In[10]:

X =  iris.drop(['Species','mapped'],axis=1)
Y =  iris[['mapped']]


# In[13]:

#KNN - K-Nearest_Neighbors
#DTC - Decision Tree Classifier
#Random Forest
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


# In[14]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y)


# In[18]:

knn.fit(xtrain,ytrain.values.ravel())


# In[19]:

ypred=knn.predict(xtest)


# In[25]:

acc = 0
for i in range (len(xtest)):
    if ytest.values[i] == ypred[i]:
        acc += 1
print(acc/len(xtest))


# In[26]:

knn.score(xtest,ytest)


# In[38]:

name=np.array(['setosa','versicolor','virginica'])


# In[39]:

type(name)


# In[41]:

name[knn.predict([[3,4.5,2.2,3.2],[5.5,6.5,7,9]])]


# In[28]:

import matplotlib.pyplot as plt
import numpy as np
accuracy = []
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain,ytrain)
    accuracy.append(knn.score(xtest,ytest))
    
plt.plot(np.arange(1,10),accuracy)
plt.show()


# In[29]:

np.arange(1,10)


# In[ ]:



