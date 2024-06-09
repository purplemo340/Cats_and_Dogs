#!/usr/bin/env python
# coding: utf-8

# ## CPE 4903 - Cats and Dogs ##

# ## HW Part I - Import data ##
# Produce the labeled data matrices X and Y by reading all 25000 images in the "train" folder. 
# * You can download the photos folders from Kaggle: https://www.kaggle.com/c/dogs-vs-cats. 
# * Download train.zip and unzip the 25000 files and move them to a folder and note the file path. For example, mine was
#   C:\Users\haiho\Dropbox\_Professor\ECE\_fall 2020\CPE4903\Animals\train 
# * You can use the functions provided in this notebook
# * hint: You can use following loop command to loop through all the files: "for i,image_file in enumerate(images) :"
# * Use train_test_split (with randomization) to split the data between train and test at 80/20. Be vigilant and verify the dimensions of X,Y throughout
# * Display the resulting data matrices that should look like the following:
#     - Shape of X_train is: (12288, 20000)
#     - Shape of X_test is: (12288, 5000)
#     - Shape of Y_train is: (1, 20000)
#     - Shape of Y_test is: (1, 5000)
# * Display the first 5 values of X_train and Y_train

# In[1]:


import os, cv2, itertools # cv2 -- OpenCV
import numpy as np 
import pandas as pd 
import time
from ipykernel import kernelapp as app
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[2]:


#Make sure do not have words "cat" or "dog" in path name
TRAIN_DIR = r'C:\Users\purpl\Downloads\animals\train\train'  
ROWS = 64
COLS = 64
CHANNELS = 3
#plt.figure(figsize=(10,20))
plt.imshow(cv2.imread(r'C:\Users\purpl\Downloads\animals\train\train\cat.0.jpg') )
a=cv2.imread(r'C:\Users\purpl\Downloads\animals\train\train\cat.0.jpg')
a.shape


# In[3]:


os.listdir(TRAIN_DIR)


# In[9]:


train_images = [TRAIN_DIR+'\\'+i for i in os.listdir(TRAIN_DIR)]
train_images1 = [TRAIN_DIR+'\\'+i for i in os.listdir(TRAIN_DIR)]


# In[5]:


y=[]
#build Y
for i,image_file in enumerate(train_images):
    if 'cat' in image_file :
        #print('cat, output = 1')
        y=np.append(y, 1)
    else:
        #print('dog, output = 0')
        y=np.append(y,0)
Y=y.reshape(25000,1)
print(y.shape)


# In[6]:


def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)


# In[10]:


# Your Part I code here. Print the shapes of all X's and Y's
#Make sure do not have words "cat" or "dog" in path name
X=[]
for i,image_file in enumerate(train_images) :
    train_images[i]=read_image(image_file)
X=np.append(X, train_images)
X = np.squeeze(X.reshape((ROWS*COLS*CHANNELS,25000)))/255
X=X.T

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 , random_state=1)
X_train=X_train.T # 12288x20000
Y_train=Y_train.T # 1x20000
X_test=X_test.T   # 12288x5000
Y_test=Y_test.T   # 1x5000
np.savez("train", X=X_train, Y=Y_train)
np.savez("valid", X=X_test, Y=Y_test)


# ## HW Part II - Manual Binary Classification Algorithm ##
# Use the equations below, which was implemented in your previous assignment of classifying the unit circle to implement the cat/dog classification using the data generated in part I. 
# <div><img src="attachment:image.png" width="300px"><div>
#     
# Your successful run of the LoR algorithm on the images should show a converging cost function similar to shown below:
# <div><img src="attachment:image-2.png" width="300px"><div>
# 
# Your train and test accuracy should be in the low 60%'s
# * Plot your cost function
# * Calculate and print your train and test accuracy (used command from previous assignment - circle)
# * Show results of classifying a new downloaded image of cat or dog (maybe correct or incorrect, try several)
# 

# In[11]:


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


# In[13]:


train=np.load('train.npz')
test =np.load('valid.npz')
X_train, Y_train= train['X'], train['Y'] 
X_test, Y_test= test['X'], test['Y'] 
print("Value Shapes")
print("Train X: ", X_train.shape, "Train Y: ", Y_train.shape)
print("Test X: ", X_test.shape, "Test Y: ", Y_test.shape)
print("Train X: ", X_train[0:5]) 
print("Train Y ", Y_train[0,0:5])
print("Test X: ", X_test[0:5]) 
print("Test Y: ", Y_test[0,0:5])
#initialize W, b, epoch
W=np.zeros((12288,1))
b=1
epoch=2000
#initialize the cost value
J = np.zeros((epoch,1))

for i in range(epoch):
    #forward propogation
    Z=np.dot(W.T,X_train)+b
    A=sigmoid(Z)
    Yhat=np.rint(A)
    L=-((Y_train* np.log(A))+((1-Y_train)* np.log(1-A)))
    J[i]=(1/20000)*np.sum(L)
    #backward propogation
    dz=A-Y_train
    dw=(1/20000)*np.dot(X_train, dz.T)
    W=W-(.0003*dw)
    db=np.sum(dz)/20000
    b=b-(.0003*db)
plt.plot(J[10:-1])
plt.title('Cost Function')
plt.show()
#Test
Z_test=np.dot(W.T,X_test)+b
A_test=sigmoid(Z_test)  
#Accuracy
acctrain = np.mean(np.rint(A) == Y_train)*100
acctest = np.mean(np.rint(A_test) == Y_test)*100
print(acctrain)
print(acctest)

# After successsful learn, with the parameters W and b, download any image of cat or dog and test your classifier using code below:
#1
file = train_images1[8]
test_image = read_image(file)
X_img = test_image.reshape(1, ROWS*COLS*CHANNELS).T/255
print(X_img.shape)
Z = np.dot(W.T, X_img) + b
A = sigmoid(Z)



if A>.5:
    print('Prediction: CAT with probability {}%'. format(A*100))
else:
    print('Prediction: DOG with probability {}%'. format(100-A*100))
plt.imshow(test_image)
plt.show()

#2
file = train_images1[24380]
test_image = read_image(file)
X_img = test_image.reshape(1, ROWS*COLS*CHANNELS).T/255
print(X_img.shape)
Z = np.dot(W.T, X_img) + b
A = sigmoid(Z)

if A>.5:
    print('Prediction: CAT with probability {}%'. format(A*100))
else:
    print('Prediction: DOG with probability {}%'. format(100-A*100))
plt.imshow(test_image)
plt.show()

#3
file = train_images1[1670]
test_image = read_image(file)
X_img = test_image.reshape(1, ROWS*COLS*CHANNELS).T/255
print(X_img.shape)
Z = np.dot(W.T, X_img) + b
A = sigmoid(Z)

if A>.5:
    print('Prediction: CAT with probability {}%'. format(A*100))
else:
    print('Prediction: DOG with probability {}%'. format(100-A*100))
plt.imshow(test_image)
plt.show()
#4
file = train_images1[76]
test_image = read_image(file)
X_img = test_image.reshape(1, ROWS*COLS*CHANNELS).T/255
print(X_img.shape)
Z = np.dot(W.T, X_img) + b
A = sigmoid(Z)

if A>.5:
    print('Prediction: CAT with probability {}%'. format(A*100))
else:
    print('Prediction: DOG with probability {}%'. format(100-A*100))
plt.imshow(test_image)
plt.show()

#5
file = train_images1[14580]
test_image = read_image(file)
X_img = test_image.reshape(1, ROWS*COLS*CHANNELS).T/255
print(X_img.shape)
Z = np.dot(W.T, X_img) + b
A = sigmoid(Z)

if A>.5:
    print('Prediction: CAT with probability {}%'. format(A*100))
else:
    print('Prediction: DOG with probability {}%'. format(100-A*100))
plt.imshow(test_image)
plt.show


# In[ ]:





# In[ ]:




