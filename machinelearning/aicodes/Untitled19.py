import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR='E:/ML_Codes/all1/train'
TEST_DIR='E:/ML_Codes/all1/test'
IMG_SIZE=50
LR = 0.0003

MODEL_NAME='dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
#pip install pyqt5
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label=='cat': return [1,0]
    elif word_label=='dog': return [0,1]
    
def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num=img.split('.')[0]
        path=os.path.join(TEST_DIR,img)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
#if u lready have train data
#train_data=np.load('train_data.csv')
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
#placeholder for input data
convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1], name='input')

convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=fully_connected(convnet,1024, activation='relu')
convnet=dropout(convnet, 0.8)

convnet=fully_connected(convnet,2,activation='softmax')
convnet=regression(convnet, optimizer='adam',
                   learning_rate=LR,
                   loss='categorical_crossentropy',
                   name='targets')

model=tflearn.DNN(convnet)

#https://pastebin.com/MYQzt1jy











    
    
    





