
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

df=pd.DataFrame({
    'x':[12,20,28,18,29,33,24,45,52,51,52,55,53,55,61,64,69,72],
    'y':[39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7]})


# In[4]:

plt.scatter(df.x,df.y)
plt.show()


# In[5]:

k=3
centroids={i+1: [np.random.randint(0,80),np.random.randint(0,80)]
          for i in range(k)}


# In[6]:

centroids


# In[8]:

plt.scatter(df['x'],df['y'],color='k')
colmap = {1: 'r',2: 'g', 3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[11]:

#Assignment Stage
def assignment(df, centroids):
    for i in centroids.keys():
        #sqrt((x1-x2^2-(y1-y2)^2))
        df['distance_from_{}'.format(i)]=(np.sqrt((df['x']-centroids[i][0]) ** 2 + (df['y']-centroids[i][1]) ** 2))
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df=assignment(df,centroids)
df.head()


# In[12]:

plt.scatter(df['x'],df['y'], color=df['color'], edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()


# In[13]:

#update Stage
import copy
old_centroids=copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)


# In[14]:

centroids


# In[15]:

old_centroids


# In[16]:

#repeat Assignment Stage
df = assignment(df, centroids)

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'], color=df['color'], edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()


# In[17]:

centroids = update(centroids)


# In[18]:

df = assignment(df, centroids)

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'], color=df['color'], edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()


# In[21]:

df=pd.DataFrame({
    'x':[12,20,28,18,29,33,24,45,52,51,52,55,53,55,61,64,69,72],
    'y':[39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7]})


# In[22]:

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3)
kmeans.fit(df)


# In[23]:

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_


# In[25]:

centroids


# In[26]:

colors = map(lambda x: colmap[x+1], labels)
colors1 = list(colors)

plt.scatter(df['x'],df['y'], color=colors1, edgecolor='k')
for idx,centroid in enumerate(centroids):
    plt.scatter(*centroid ,color=colmap[idx+1])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()


# In[27]:

cost =[] 
for i in range(1, 11): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(df)     
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)  


# In[28]:

# plot the cost against K values 
plt.plot(np.arange(1, 11), np.array(cost)) 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 


# In[ ]:



