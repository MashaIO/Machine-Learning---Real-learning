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
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))

#%%
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])



#%%

#%%

#%%
total_training_data[0][1]

#%%
from sklearn.model_selection import train_test_split

#%%
print(type(total_training_data))
print(total_training_data.shape)
print(total_training_data.size)
print(total_training_data[0])
print(total_training_data[0][0])
print(total_training_data[0][1])

#%%
train_X = []
for x in range(1600):
  train_X.append(total_training_data[x][0])

#%%
print(train_X[0])

#%%
train_Y_one_hot = []
for x in range(1600):
  train_Y_one_hot.append(total_training_data[x][1])

#%%
print(train_Y_one_hot)


#%%
#
# train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
#train_X_data,valid_X_data,train_label_data,valid_label_data = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

#%%
train_X.reshape([-1,28, 28,1])



#%%
#model_train_dropout = model.fit(train_X, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X_data, valid_label_data))
model_train_dropout = model.fit(np.array(train_X), np.array(train_Y_one_hot), batch_size = 50, epochs = 5, verbose = 1)

