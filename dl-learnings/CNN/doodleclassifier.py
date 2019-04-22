#%%
import numpy as np

#%%
# Ther are totally 123202 images with each having 784 pixel 
# with this we can easily find the height and width
# Hope eveeyone understands what is the difference between size and shape. :-)

catsnpdata = np.load("data\cat.npy")


#%%
print(catsnpdata.shape)
print(catsnpdata.size)
print(type(catsnpdata))

#%%


#%%
arr = catsnpdata[0].reshape(28, 28)
arr.shape
plt.imshow(arr)


#%%
cats_Data = np.fromfile("data\cat1000.bin", dtype='uint8')
clocks_Data = np.fromfile("data\clock1000.bin", dtype='uint8')

#%%
print(type(cats_Data))
print(cats_Data.shape)
print(clocks_Data.shape)


#%%
# for cat in cats_data:
#   print(cat)
for x in range(200):
  print(cats_Data[x])

#%%
type(cats_Data)

#%%
cats_Data = cats_Data.reshape(-1, 28,28, 1)

#%%
# -1 is to tell total lenght and 1 is number of channels
clocks_Data = clocks_Data.reshape(-1, 28,28, 1)

#%%
print(cats_Data.shape)
print(clocks_Data.shape)


#%%
cats_Data = cats_Data / 255.
clocks_Data = clocks_Data / 255.

#%%
cats_Data.size

#%%
for x in range(200):
  print(cats_Data[x])


#%%
import pandas as pd
cats_training_data = []
cats_test_data = []
catlabel = 1

# Lets split the data.
for x in range(1000):
  offset = x * 784
  endOfIndex = offset + 784
  #print(offset)
  #print(endOfIndex )
  if(x < 800):
    cats_training_data.append([np.array(cats_Data[offset : endOfIndex]), np.array([1, 0])])
  else:
    cats_test_data.append([np.array(cats_Data[offset : endOfIndex]), np.array([1, 0])])

#%%
print(type(cats_training_data[799]))
print(cats_training_data[799])
print(cats_test_data[199])

#%%
clock_training_data = []
clock_test_data = []
clock_label = 3

# Lets split the data.
for x in range(1000):
  offset = x * 784
  endOfIndex = offset + 784
  #print(offset)
  #print(endOfIndex )
  if(x < 800):
    clock_training_data.append([np.array(clocks_Data[offset : endOfIndex]), np.array([0, 1])])
  else:
    clock_test_data.append([np.array(clocks_Data[offset : endOfIndex]), np.array([0, 1])])


#%%
print(clock_training_data[799])
print(cats_test_data[199])

#%%
print(type(clock_training_data))
print(type(clock_training_data))


#%%
total_training_data = np.concatenate((cats_training_data, clock_training_data), axis=0)


#%%
print(total_training_data.shape)
print(total_training_data.size)



#%%
np.random.shuffle(total_training_data)

#%%
total_training_data[0]

#%%
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU



#%%
batch_size = 64
epochs = 20
num_classes = 2

#%%
model = Sequential()
