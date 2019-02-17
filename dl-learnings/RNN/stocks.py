#%%
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%
dataset = pd.read_csv('data/Google.csv')
dataset.head()
len(dataset)

#%%
# Lets split the data.
from sklearn.model_selection import train_test_split


#%%
stock_opening_Values = dataset.iloc[:,1:2].values
stock_opening_Values
x = stock_opening_Values
print (x)

#%%
stocl_date_values = dataset.iloc[:,0:1].values
stocl_date_values
y = stocl_date_values
print (y)


#%%
# Lets split the data.
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=1/3, random_state=0)

#%%
print (x_train)

#%%
print(x_test)

#%%
print(y_train)

#%%
print(y_test)

#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))

# the above lines to convert the google.csv to 'Open' values 
# from 0 to 1 

#%%
print(sc)


#%%
# feature scaling.
x_train_scaled_as_one_and_zero = sc.fit_transform(x_train)
len(x_train_scaled_as_one_and_zero)

#%%
print (x_train_scaled_as_one_and_zero)


#%%
xx_train = []
yy_train = []

for i in range(60, 2083):
 
    # Here it will be always the 'Open' values do you know why?
    # we have already filtered

    # [60 - 60 : 60 , 0]
    # [0:60, 0]

    # X = 0 - 59 and Y = 60

    # [61 - 60 : 60 , 0]
    # [1:60, 0]

    # X = 1 - 60 and Y = 61

    # [62 - 60 : 60 , 0]
    # [2:60, 0]

    # X = 2 - 61 and Y = 62

    xx_train.append(x_train_scaled_as_one_and_zero[i-60:i, 0])
    yy_train.append(x_train_scaled_as_one_and_zero[i, 0])

#%%
print("xx_train:: ") 
print(xx_train)

print("yy_train:: ")
print(yy_train)
#print(i)

#%%
xx_train, yy_train = np.array(xx_train), np.array(yy_train)

#%%
# Re shapping

xx_train = np.reshape(xx_train, (xx_train.shape[0], xx_train.shape[1], 1))
print(xx_train)


#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#%%
regressor = Sequential()

#%%
# First LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (xx_train.shape[1],1)))
regressor.add(Dropout(0.2))

#%%
# Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#%%
# Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#%%
# Fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#%%
# output layer
regressor.add(Dense(units = 1))

#%%
regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')

#%%
# fitting 
regressor.fit(xx_train, yy_train, epochs = 100, batch_size = 32)

