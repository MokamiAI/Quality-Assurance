#!/usr/bin/env python
# coding: utf-8

# ###### Image Outlier Detection using Artificial Neural Network
# 
# ## Introduction
# 
# Artificial intelligence is commonly used in various trade circles to automate processes, gather insights on business, and speed up processes. 
# 
# In Machine Learning, anomaly detection tasks are quite common. Data Scientists are frequently engaged in problems where they have to show, explain and predict anomalies. Detecting anomalies and predicting them beforehand can save a large amount of money. As always, AI can help us in this case. 
# 
# In this Project, I focused on image outlier detection using artificial neural networks.
# 
# ## Context
# 
# I will be working with dataset showing perfect walls and cracked walls, obtained from [Kaggle](https://www.kaggle.com/rahatreza/notebook361c281ea4/data). Half of the images in the dataset show new and uncorrupted pieces of the wall; the remaining part shows cracks of various dimensions and types.
# 
# 
# ### Side note: What is VGG?
# 
# VGG is a neural network used for large scale image recognition. I will heavily use VGG in this notebook.It was trained on the ImageNet database which has over 14 million hand annotated images.
# 
# 
# ## Use Python to open csv files
# 
# I will use the [scikit-learn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/), [TensorFlow](https://www.tensorflow.org/), and [Keras](https://keras.io/) libraries to work with our dataset. Scikit-learn is a very useful machine learning library that provides efficient tools for predictive data analysis. TensorFlow and Keras are open source libraries that help in developing and training ML models. Pandas is a popular Python library for data science. It offers powerful and flexible data structures to make data manipulation and analysis easier.
# 
# 
# ## Import Libraries
# 

# In[1]:


import numpy as np 
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.utils import *

from PIL import Image
import requests
from io import BytesIO
import os
import random
import pickle
import tqdm
import itertools

import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ## Reading the Dataset
# 
# I defined the train and test directories here. I used a large dataset for the train and test data even though it can be used on small datsets.
# 
# 
# The train directory has 2 subdirectories:
# 
# - cracked(folder contains the cracked wall images)
# 
# 
# - uncracked(folder contains the uncracked wall images)

# In[3]:


# define training and test data directories
Train_dir = 'TrainReducedQualityAssuranceSystem/'

#Loading the test dataset
test_dir = 'TestQualityAssuranceSystem/'


# classes are folders in each directory with these names
classes = ['cracked','uncracked']


# In[5]:


#our artificial neural network processes images of a particular size 224X224.
#So , I will transform our images first
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)



# In[31]:


print('Num of training Images:', len(train_data))


# In[32]:


# define  parameters
batch_size = 32
num_workers=0

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                         num_workers=num_workers, shuffle=True)


# In[ ]:


#displaying some of the images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])


# In[33]:


#Let us initialize the model
vgg16 = models.vgg16(pretrained=True)

# print out the model structure
print(vgg16)


# In[34]:


#printing the number of input and output features
print(vgg16.classifier[6].in_features) 
print(vgg16.classifier[6].out_features)


# 
# ## Training
# 
# I will use the VGG model to train our dataset so that I can identify anomalies in the test data. I will use the *softmax* function and *categorical_crossentropy* loss function for this.
# 
# The *softmax function* is an exponential function and is used in classification and regression methods. Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes. For more information, see [Softmax](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)
# 
# Categorical crossentropy is a loss function that is used in classification tasks. Loss function is a measure of how good the classification is. The lower is the loss, the better is the classification. For more details, check [loss](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy)
# 
# 
# One *Epoch* is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. *Batch size* is the total number of training examples present in a single batch.

# In[35]:


#The VGG model has many different layers - convolutional layers, max pooling layers and 
#dense layers. We do not need to take all of them for our computation. 

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False

n_inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
vgg16.classifier[6] = last_layer


# In[36]:


import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.005)


# 
# 
# ### Training
# 

# In[37]:


n_epochs = 60
larr = []

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # model by default is set to train
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        
        if epoch % 5 == 0:    # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 32))
            larr.append(train_loss / 32)
            train_loss = 0.0


# In[ ]:


#plot loss
plt.plot(larr)

