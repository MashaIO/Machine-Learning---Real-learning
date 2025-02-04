#%%
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


#%%
dataset_train = pd.read_csv("data/stocks/trainset.csv")
dataset_train.head()




#%%
len(dataset_train)

#%%
trainset = dataset_train.iloc[:,1:2].values
trainset


#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)
training_scaled

#%%
x_train = []
y_train = []
for i in range(60,1259):
    x_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)

#%%
x_train.shape

#%%
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#%%
regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

# regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)
# regressor.save('model/google_stocks_sep_train_test.h5')

#%%
from keras.models import load_model
regressor = load_model('model/google_stocks.h5')

#%%
dataset_test =pd.read_csv("data/stocks/testset.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
dataset_total

#%%
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs



#%%
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
inputs.shape

#%%
x_test = []
for i in range(60,185):
    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)
x_test.shape

#%%
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape

#%%
predicted_price = regressor.predict(x_test)

#%%
predicted_price = sc.inverse_transform(predicted_price)
predicted_price

#%%
plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()