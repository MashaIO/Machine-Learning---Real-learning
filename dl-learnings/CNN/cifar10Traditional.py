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
# Step 1: FITERS:
# Filters the unwanted pixels and forms smaller matrix and gives the features
# Step 2: Relu Layer
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
# Step 4  Flattening 
# 6
# 8
# 4 
# 7

# Step 5 Fully connected layer
# Here the image classification happens

# Lets code
#%%
import os

#%% [markdown]
# 
# Let's use CIFAR-10 dataset
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
# with 6000 images per class. 
# There are 50000 training images and 10000 test images. 
# The dataset is divided into five training batches and one test batch, each with 10000 images. 
# The test batch contains exactly 1000 randomly-selected images from each class. 
# The training batches contain the remaining images in random order, 
# but some training batches may contain more images from one class than another. 
# Between them, the training batches contain exactly 5000 images from each class. 
# 
#%% [markdown]
# Step 0: Get the Data
#%%
# Put file path as a string here
CIFAR_DIR = 'C:/Tulip/Machine-Learning---Real-learning/data/cifar-10-batches-py/'

#%% [markdown]
# The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. 
# Each of these files is a Python "pickled" object produced with cPickle. 
# 
# ** Load the Data. Use the Code Below to load the data: **
#%%
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

# 60000 
#%%
dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']


#%%
all_data = [0,1,2,3,4,5,6]


#%%
for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)


#%%
batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]


#%%
batch_meta
# CHeck for images
data_batch1

#%% [markdown]
# ** Why the 'b's in front of the string? **
# Bytes literals are always prefixed with 'b' or 'B'; 
# they produce an instance of the bytes type instead of the str type. 
# They may only contain ASCII characters; 
# bytes with a numeric value of 128 or greater must be expressed with escapes.
# https://stackoverflow.com/questions/6269765/what-does-the-b-character-do-in-front-of-a-string-literal

#%%
data_batch1.keys()

#%% [markdown]
# Loaded in this way, each of the batch files contains a dictionary with the following elements:
# * data -- a 10000x3072 numpy array of uint8s. 
# Each row of the array stores a 32x32 colour image. 
# The first 1024 entries contain the red channel values, the next 1024 the green, 
# and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# * labels -- a list of 10000 numbers in the range 0-9. 
# The number at index i indicates the label of the ith image in the array data.
# 
# The dataset contains another file, called batches.meta. 
# It too contains a Python dictionary object. It has the following entries:
# 
# * label_names -- a 10-element list which gives meaningful names to the numeric labels
#  in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
#%% [markdown]
# ### Display a single image using matplotlib.
# 
# ** Grab a single image from data_batch1 and 

#%%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np


#%%
X = data_batch1[b"data"] 

# Out of 10,000 images 32 * 32 picture and 3 bits 
# 10,000 images are broke down into 3 pieces
# RGB
# Transpose ( 0 - image , 2 is one 32 , 3 is another 32 and 1 is 3 pieces)
# "unit8" - Ram size
#%%
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

# 0 to 255 float values
#%%
X[0].max()

#%%
plt.imshow(X[4])

#%%
(X[0]/255).max()


#%%
plt.imshow(X[45])


#%%
plt.imshow(X[50])

#%% [markdown]
# # Helper Functions for Dealing With Data.
# 
# ** Use the provided code below to help with dealing with grabbing the
#  next batch once you've gotten ready to create the Graph Session. 
# 10 possible lables , 
# Which denotes data of car or dog [0,1,1,0,1,*****]

#%%
def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


#
#%%
class CifarHelper():
    # Initializing variables
    def __init__(self):
        self.i = 0
        
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        self.test_batch = [test_batch]
        
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        # filling the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        #
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print("Setting Up Test Images and Labels")
        
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    #  
    def next_batch(self, batch_size):
        # Just first 100 images
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        # x is the image and y is the label
        return x, y

#%% [markdown]
# ** How to use the above code: **

#%%
# Before Your tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()


# batch = ch.next_batch(100)

#%% [markdown]
# ## Creating the Model
# 
# ** Import tensorflow **

#%%
import tensorflow as tf

#%% [markdown]
# ** Create 2 placeholders, x and y_true. Their shapes should be: **
# 
# * x shape = [None,32,32,3]
# * y_true shape = [None,10]
# 

#%%
x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,10])

#%% [markdown]
# ** Create one more placeholder called hold_prob. No need for shape here. 
# This placeholder will just hold a single probability for the dropout. **

#%%
hold_prob = tf.placeholder(tf.float32)

#%% [markdown]
# ### Helper Functions
# 
# ** Grab the helper functions from MNIST with CNN (or recreate them here yourself for a hard challenge!). You'll need: **
# 
# * init_weights
# * init_bias
# * conv2d
# * max_pool_2by2
# * convolutional_layer
# * normal_full_layer

# 
#%%
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    # reducing 32 * 32 * 3 into 2D
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# getting the max value
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

# fully connected layer inuput will be flatten
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

#%% [markdown]
# ### Create the Layers
# 
# ** Create a convolutional layer and a pooling layer as we did for MNIST. **
# ** Its up to you what the 2d size of the convolution should be, but the last two digits need to be 3 and 32 because of the 3 color channels and 32 pixels. So for example you could use:**
# 
#         convo_1 = convolutional_layer(x,shape=[4,4,3,32])

# STEP 1
#%%
# 3 channels , 32 - pixels each, 4 - filter size, 4 filter size
convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling = max_pool_2by2(convo_1)

#%% [markdown]
# ** Create the next convolutional and pooling layers.  The last two dimensions of the convo_2 layer should be 32,64 **

#%%
convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

#%% [markdown]
# ** Now create a flattened layer by reshaping the pooling layer into [-1,8 \* 8 \* 64] or [-1,4096] **

# STEP 2 
# 8*8*64 bytes 4096
#%%
convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64])

#%% [markdown]
# ** Create a new full layer using the normal_full_layer function and
#  passing in your flattend convolutional 2 layer with size=1024. (You could also choose to reduce this to something like 512)**

# STEP 3 
#%%
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

#%% [markdown]
# ** Now create the dropout layer with tf.nn.dropout, 
# remember to pass in your hold_prob placeholder. **

#%%
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

#%% [markdown]
# ** Finally set the output to y_pred by passing in the dropout layer into the normal_full_layer function. The size should be 10 because of the 10 possible labels**

# 10 labels
#%%
y_pred = normal_full_layer(full_one_dropout,10)
y_pred

#%% [markdown]
# ### Loss Function
# 
# ** Create a cross_entropy loss function **
# Improve gain 
# Magic happens heres . Labels is nothing but dog or car
#%%
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

#%% [markdown]
# ### Optimizer
# ** Create the optimizer using an Adam Optimizer. **

# redues the entropy
#%%
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

#%% [markdown]
# ** Create a variable to intialize all the global tf variables. **

#%%
init = tf.global_variables_initializer()

#%% [markdown]
# ## Graph Session
# 
# ** Perform the training and test print outs in a Tf session and run your model! **
saver = tf.train.Saver()


#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 500 * 100 = ?
    for i in range(500):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_prob:1.0}))
            print('\n')
    
    saver.save(sess, 'C:/Tulip/Machine-Learning---Real-learning/model/my_cifar_model')
    saver.save(sess, 'cifar_tf_model')










