
# coding: utf-8

# In[2]:


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[58]:


dataset = pd.read_csv('data/Google.csv')
dataset.head()
len(dataset)


# In[59]:


# Lets split the data.
from sklearn.model_selection import train_test_split


# In[61]:


stock_opening_Values = dataset.iloc[:,1:2].values
stock_opening_Values
x = stock_opening_Values
print (x)


# In[62]:


stocl_date_values = dataset.iloc[:,0:1].values
stocl_date_values
y = stocl_date_values
print (y)


# In[63]:


# Lets split the data.
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=1/3, random_state=0)


# In[64]:


print (x_train)


# In[65]:


print(x_test)


# In[66]:


print(y_train)


# In[67]:


print(y_test)


# In[68]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))

# the above lines to convert the google.csv to 'Open' values 
# from 0 to 1 


# In[13]:


print(sc)


# In[69]:


# feature scaling.
x_train_scaled_as_one_and_zero = sc.fit_transform(x_train)
len(x_train_scaled_as_one_and_zero)


# In[70]:


print (x_train_scaled_as_one_and_zero)


# In[71]:


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


# In[72]:


print("xx_train:: ") 
print(xx_train)

print("yy_train:: ")
print(yy_train)
#print(i)


# In[73]:


xx_train, yy_train = np.array(xx_train), np.array(yy_train)


# In[19]:


# Re shapping

xx_train = np.reshape(xx_train, (xx_train.shape[0], xx_train.shape[1], 1))
print(xx_train)


# In[74]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# In[75]:


regressor = Sequential()


# In[22]:


# First LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (xx_train.shape[1],1)))
regressor.add(Dropout(0.2))


# In[23]:


# Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# In[24]:


# Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# In[25]:


# Fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[26]:


# output layer
regressor.add(Dense(units = 1))


# In[27]:


regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')


# In[28]:


# fitting 
#regressor.fit(xx_train, yy_train, epochs = 100, batch_size = 32)


# In[29]:


x_test


# In[30]:


# regressor.save('model/google_stocks.h5')


# In[31]:


dataset_open = dataset['Open']
print(dataset_open)


# In[32]:


# Lets predict
print(len(dataset_open))
print(len(x_test))
print(len(dataset_open) - len(x_test) - 60)

inputs = dataset_open[len(dataset_open) - len(x_test) - 60: ].values


# In[33]:


print(inputs)


# In[34]:


inputs = inputs.reshape(-1, 1)
print(inputs)


# In[35]:


inputs = sc.transform(inputs)


# In[36]:


print(inputs)


# In[37]:


len(x_test)


# In[83]:


xxx_test = []
for i in range(60,1042):
    xxx_test.append(inputs[i-60:i, 0])


# In[84]:


xxx_test = np.array(xxx_test)
xxx_test = np.reshape(xxx_test, (xxx_test.shape[0], xxx_test.shape[1], 1))
#print(xxx_test)


# In[85]:


from keras.models import load_model
model = load_model('model/google_stocks.h5')


# In[87]:


predicted_stock_price = model.predict(xxx_test)


# In[88]:


predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[81]:


print(predicted_stock_price)
print(x_test)

plt.plot(x_test, color = 'red', label = 'Actual Google stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted google stock price')
plt.title('Google Stock Price prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()