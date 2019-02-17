#%%
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%
dataset = pd.read_csv('data/Google.csv')
dataset.head()

#%%
# Lets split the data.
from sklearn.model_selection import train_test_split


#%%
stock_opening_Values = dataset_train.iloc[:,1:2].values
stock_opening_Values
x = stock_opening_Values
print (x)

#%%
stocl_date_values = dataset_train.iloc[:,0:1].values
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
print (x_train_scaled_as_one_and_zero)



