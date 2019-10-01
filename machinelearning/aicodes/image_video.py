
# coding: utf-8

# In[1]:


import cv2,time
video=cv2.VideoCapture(0)


# In[2]:


state,frame=video.read()
print(state)
print(frame)
time.sleep(3)
cv2.imshow('my_image',frame)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()


# In[ ]:


video=cv2.VideoCapture(0)
a=1
while True:
    a=a+1
    check, frame = video.read()
    print(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('capturing',gray)
    key=cv2.waitKey(1)
    if key == ord('a'):
        break
        
print(a)
video.release()
cv2.destroyAllWindows()


# In[9]:


#Face Recognition
import cv2
import sys


# In[10]:


imagepath='E:/ML_Codes/index.jpg'
cascpath='E:/ML_Codes/FaceDetect-master(1)/FaceDetect-master/haarcascade_frontalface_default.xml'


# In[11]:


#creating a face classification
faceCascade=cv2.CascadeClassifier(cascpath)


# In[12]:


#Read the image
image=cv2.imread(imagepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[13]:


#Face Detection
faces=faceCascade.detectMultiScale(image, scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(30,30),
                                  flags=cv2.CASCADE_SCALE_IMAGE)


# In[14]:


print(len(faces))
for (x ,y, w, h) in faces:
    cv2.rectangle(image, (x,y), 
                  (x+w, y+h), (0,255,0),2)

cv2.imshow('Faces Found', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




