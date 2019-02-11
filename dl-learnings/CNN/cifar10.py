# CNN
# Image -> Input Layer > Hidden Layer - Output Layer
# 
# Input Layer :
#  Accepts the images as part of pixels and form arrays
#
# Hidde Layer
# Feature Extraction
#    - Convolution Layer (சுழற்சி)
#    - Relu layer
#    - Pooling layer 
#    - Fully connected layer.

#  
#         CL        RL        PL
#   IL      O        O         O
#   O       O        O         O
#   O       O        O         O
#   O       O        O         O
#   O       O        O         O
#
#
#  Images will be converted as matrix (Assume white space as 0
#                                          and dark place as 1) 
# 
#  a= [5,3,2,5,9,7]
#  b= [1,2,3]
#  a * b = [5*1, 3*2, 2*3 ] = Sum = 17
#          [3*1, 3*2, 3*2 ] = Sum = 22
# Final matrix  - [17 ,22, **]
# Step 1: FILTERS:
# Filters the unwanted pixels and forms smaller matrix and gives the features

# Step 2: RELU Layer
# Skip the negative values. 
# Gives multiple features and muliple relu layers

# Step 3: Pooling(Edges)
# Down sampling and will give smaller dimensions

#
#Rectified Fetaure map
# 1 4 2 7
# 2 6 8 5
# 3 4 0 7
# 1 2 3 1

# Arriving the max value 
# 6 8
# 4 7

# Finally Getting the 2 dimensional 

# Step 4  Flattening - Gives the ouput as Vector
# 6
# 8
# 4 
# 7

# Step 5 Fully connected layer
# Here the image classification happens (Signmoid or softmax)

# Lets work Cifar 10 data set.

# Lets code
#%%
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import print_summary, to_categorical
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
from matplotlib import pyplot
from scipy.misc import toimage


#%%
batch_size = 64
num_classes = 10
epochs = 10
model_name = 'keras_cifar10_model'
save_dir = 'model/' + model_name

# The data, split between train and test sets:
#%%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
#%%
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#%%
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

#%%
model = Sequential()

#  padding='same' - Drop the part of the image where the filter did not fit. 
#  This is called valid padding which keeps only valid part of the image.
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

# ReLU’s purpose is to introduce non-linearity in our ConvNet. Since, the real world data would want our ConvNet to learn would be non-negative linear values.
# ƒ(x) = max(0,x).
# relu or tanh or sigmoid can be used.
model.add(Activation('relu'))

# Pooling layers section would reduce the number of parameters when the images are too large.
# Spatial pooling also called subsampling or downsampling which reduces the dimensionality of each map but retains the important information. Spatial pooling can be of different types:

# Max Pooling
# Average Pooling
# Sum Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Entropy 
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(80))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes))

# Can be sigmoid or Softmax IMAGE CLASSIFICATION
model.add(Activation('softmax'))

#SHopastic Gradi
# initiate RMSprop optimizer
#%%
opt = SGD(lr=0.01, momentum=0.9, decay=0, nesterov=False)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

#%%
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True)

#%%
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#%%
# Lets test our model.
model = load_model('model/keras_cifar10_model.h5')
model


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Visualize the image.
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    print(y_train[i])
    pyplot.imshow(X_train[i])
# show the plot
pyplot.show()

for i in range(10, 19):
    pyplot.subplot(330 + 1 + i)
    print(y_train[i])
    pyplot.imshow(X_train[i])
# show the plot
pyplot.show()

for i in range(10, 19):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i])
# show the plot
pyplot.show()

for i in range(10, 19):
     print(y_train[i])

# Simply function to get the class name 
def load_label_names(predictedLableValue):
    dictionary = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    for classname, labelvalue in dictionary.items():
         if labelvalue == predictedLableValue:
            print(classname)

img = cv2.imread('images/cat32by32.png')

plt.imshow(img)
plt.show()
img.shape

img_newcopy = cv2.imread('images/cat32by32.png')
img_small = cv2.resize(img_newcopy,(32,32))

plt.imshow(img_small)
plt.show()
img_small.shape


img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)

img_small = cv2.imread('images/truck1.png')
img_small = cv2.resize(img_small,(32,32))
img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)

img_small = cv2.imread('images/truck2.png')
img_small = cv2.resize(img_small,(32,32))
img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)

img_small = cv2.imread('images/airplane1.png')
img_small = cv2.resize(img_small,(32,32))
img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)

img_small = cv2.imread('images/airplane2.png')
img_small = cv2.resize(img_small,(32,32))
img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)

img_small = cv2.imread('images/bird1.png')
img_small = cv2.resize(img_small,(32,32))
img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)

img_small = cv2.imread('images/bird2.png')
img_small = cv2.resize(img_small,(32,32))
img_small = np.reshape(img_small,[1,32,32,3])
classes = model.predict_classes(img_small)
load_label_names(classes)
