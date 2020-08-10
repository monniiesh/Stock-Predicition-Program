#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the Stock Quote
df = web.DataReader('ITC.NS',data_source='yahoo',start='2012-01-01',end='2020-4-19')

#Get the Number of Rows and Columns in the data set
print(df.shape)

# Visualize the closing price history

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Prize',fontsize=18)
plt.show()Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Prize',fontsize=18)
plt.show()

#Create a new dataframe with only the 'Close Column'
data = df.filter(['Close'])
#convert the dateframe to a numpy array
dataset = data.values
#Get the Number of Rows to train the model on
training_data_len = math.ceil(len(dataset))
print(training_data_len)

#Scale the Data
scaler = MinMaxScaler(feature_range=(0,1)) #makes the comparison into binary fro the LSTM Model
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)


#Create the Training Dataset
#Create the scaled training data set
train_data = scaled_data[0:training_data_len,:]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])

#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the Data (LSTM expects the data to be in 3-Dimensional [No. of samples, No. of features, No. of time steps])
#x_train.shape = (1543,60)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))   # np.reshape(dataset, (1st dimension[rows], 2nd dimension[columns], 3rd dimesnsion [close day]))
print(x_train.shape)
#print(x_train)

#Build LSTM Model
model = Sequential()
#1st LSTM Layer
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1))) #LSTM(No. of Neurons, Want another LSTM layer, 1st layer shape(features, time step))
#2nd LSTM Layer
model.add(LSTM(50, return_sequences=False)) #We dont want another LSTM layer
#3rd Dense Layer
model.add(Dense(25)) #25 neurons
#4th Dense Layer
model.add(Dense(1)) # 1 neuron

#Compile the Model
model.compile(optimizer = 'adam', loss = 'mean_squared_error') # optimizer(imporves the loss function), loss(tell how well the model did on training)

#Train the Model
model.fit(x_train, y_train, batch_size = 1, epochs=1) #fit[train](x_train,y_train,total no. of training example present in a single batch,no. of iterations passed forward and backward through a neural network)

#Create the Testing Dataset
#Create a new array containing scaled values from index 1543 to 2003(the end of the dataset)
test_data = scaled_data[training_data_len - 60: , :]
#Create the Datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:,:] #all the values we want our data to predict
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#Covert the Data into Numpy Array
x_test = np.array(x_test)
print(x_test.shape)

#Reshape the Data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get the models Predicted Prize Values
predictions = model.predict(x_test)
#Inverse transform the data to match the no. of y_test datasets (Unscaling the Values)
predictions = scaler.inverse_transform(predictions)

#Get the Root mean squared Error (RMSE) - Good measure of how accurate a model predicts/ standard deviation of the residuals/lowaer values of RMSE depict a better fit
rmse = np.sqrt(np.mean(((predictions-y_test)**2)))
print(rmse) #if rmse is 0, then the predictions are exctly right

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the Data
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close  Price', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val', 'Predictions'])
plt.show()
