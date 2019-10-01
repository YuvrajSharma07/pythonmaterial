
# coding: utf-8

# In[1]:

import os
import numpy as np
def load_dataset():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("downloading",filename)
        import urllib.request
        #from urllib.request import urlretrieve
        urllib.request.urlretrieve(source+filename,filename)
        
    import gzip
        
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
                
        with gzip.open(filename,'rb') as f:
            data=np.frombuffer(f.read(), np.uint8, offset=16)
                
            data=data.reshape(-1,1,28,28)
                
            return data/np.float32(256)
            
            
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
                    
        with gzip.open(filename,'rb') as f:
            data=np.frombuffer(f.read(), np.uint8, offset=8)
                        
            return data
            
    X_train=load_mnist_images('train-images-idx3-ubyte.gz')
    y_train=load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test=load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test=load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    return X_train, y_train, X_test, y_test



X_train, y_train, X_test, y_test = load_dataset()



# In[ ]:

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.show(plt.imshow(X_train[4][0]))


# In[2]:

import lasagne
import theano
import theano.tensor as T


# In[3]:

def build_NN(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    l_in_drop=lasagne.layers.DropoutLayer(l_in,p=0.2)
    
    l_hid1=lasagne.layers.DenseLayer(l_in_drop,num_units=800,nonlinearity=lasagne.nonlinearities.rectify,
                                     W=lasagne.init.GlorotUniform())
    
    l_hidl1_drop=lasagne.layers.DropoutLayer(l_in,p=0.5)
    
    
    l_hid2=lasagne.layers.DenseLayer(l_hidl1_drop,num_units=800,nonlinearity=lasagne.nonlinearities.rectify,
                                     W=lasagne.init.GlorotUniform())
    
    l_hidl2_drop=lasagne.layers.DropoutLayer(l_hid2,p=0.5)
    
    l_out=lasagne.layers.DenseLayer(l_hidl2_drop,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    
    return l_out
    
input_var=T.tensor4('inputs')  #placeholders
    
target_var=T.ivector('targets')
    
network=build_NN(input_var)
    
prediction=lasagne.layers.get_output(network)
loss=lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss=loss.mean()
    
params=lasagne.layers.get_all_params(network, trainable=True)
updates=lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01,momentum=0.9)
    
train_fn=theano.function([input_var, target_var], loss, updates=updates)


# In[ ]:

num_training_steps=10
for step in range(num_training_steps):
    train_err=train_fn(X_train, y_train)
    print("Current step is "+str(step))


# In[ ]:

test_prediction=lasagne.layers.get_output(network)
val_fn=theano.function([input_var],test_prediction)

val_fn([X_test[0]])


# In[ ]:

y_test[0]


# In[ ]:

test_prediction=lasagne.layers.get_output(network)
test_acc=T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var),dtype=theano.config.float)
acc_fn=theano.function([input_var,target_var],test_acc)

acc_fn(X_test,y_test)


# In[ ]:



