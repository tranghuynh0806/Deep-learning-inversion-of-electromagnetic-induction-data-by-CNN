# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:01:18 2021

@author: huyn389
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from numpy.random import seed
from tensorflow.python.framework.random_seed import set_random_seed
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import joblib
import csv

# Open and read the file of model, bz_imaginary, bz_real


with open('C:/Users/huyn389/OneDrive - PNNL/Desktop/model_25000.csv', 'r') as csvfile_model:
    reader_model = csv.reader(csvfile_model)
    rows_model = [row for row in reader_model]
    data_model = np.array(rows_model[::2]) # only read every other rows
    data_model = np.asarray(data_model, dtype=np.float64, order='C')
# No read blank space

with open ('C:/Users/huyn389/OneDrive - PNNL/Desktop/bz_25000.csv', 'r') as csvfile_bz:
    reader_bz = csv.reader(csvfile_bz)
    rows_bz = [row for row in reader_bz]
    data_bz = np.array(rows_bz[::2])
    data_bz = np.asarray(data_bz, dtype=np.float64, order='C')
#print("HERE", rows_bz) 


### function to split data
# 70% training, 15% validation, 15% test

# Splitting the input data
def data_split_seq(data, N, pr_train, pr_test, pr_validate):
    ### N :  total number of models
    pr_train = pr_train/100 # get the percentage of training
    pr_test = pr_test/100 # get the percentage of testing
    pr_validate = pr_validate/100
    N_train = int(N * pr_train) # calculate the training number
    N_test = int(N * pr_test) # calculate the testing number
    N_validate = int(N* pr_validate)
    index_train = np.arange(0, N_train,1) # index_train is 0 to 14000
    index_test = np.arange(0, N_test, 1) # index_test is 0 to 3000
    index_validate = np.arange(0, N_validate, 1)
    train = data[index_train][:]
    test1 = np.delete(data, index_train, axis=0)
    #test = test1[index_test, :] # all the columns of 3000 rows
    test = test1[index_test][:]
    validate1 = np.delete(test1, index_test, axis=0)
    validate = validate1[index_validate][:]
    return train, validate, test



## function to reshape input data
def input_prep(X, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X
##### function to reshape output data
def output_prep(Y, n_features):
    Y = Y.reshape((Y.shape[0], Y.shape[1], n_features))
    return Y

## Split data into training validation and test subsets
input_train, input_val, input_test = data_split_seq(data_bz, len(data_bz), 70, 15, 15) # 70% training and 15% testing
output_train, output_val, output_test = data_split_seq(data_model, len(data_model), 70, 15, 15)

## data normalization
scaler_in = MinMaxScaler(feature_range=(0, 1))
scaler_out = MinMaxScaler(feature_range=(0, 1))

scaler_in_T = scaler_in.fit(input_train)

scaler_out_T = scaler_out.fit(output_train)

input_train_norm = scaler_in.transform(input_train)
input_val_norm = scaler_in.transform(input_val)
input_test_norm = scaler_in.transform(input_test)

output_train_norm = scaler_out.transform(output_train)
output_val_norm = scaler_out.transform(output_val)
output_test_norm = scaler_out.transform(output_test)

### number of input and outputs and features
n_features = 1
n_input = 10
n_output = 20
#n_input, n_output = data_bz.shape[0][:], data_model.shape[0][:]



### make input
input_train_norm = input_prep(input_train_norm, n_features)
input_val_norm = input_prep(input_val_norm, n_features)
input_test_norm = input_prep(input_test_norm, n_features)

############# make output
output_train_norm = output_prep(output_train_norm, n_features)
output_val_norm = output_prep(output_val_norm, n_features)
output_test_norm = output_prep(output_test_norm, n_features)

## Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

### set seed
## for keras
seed(0)
## for tensorflow
set_random_seed(0)

### define model: Sequential model are linear stacks of layers where one 
### layer leads to the next.  Have to make sure the previous layer is the
### input to the next layer
model = keras.Sequential()

## leakage of the leaky relu
LRU = 0.01

model.add(keras.layers.Conv1D(filters=5, kernel_size=2, strides=1, padding="same", input_shape=(n_input, n_features)))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=10, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=20, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=40, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=80, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=160, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=320, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
model.add(keras.layers.Conv1D(filters=640, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=1))
### The result after first convolutional layer is 2*640



### the second set of convolutional layers
# play around the number of filter

model.add(keras.layers.Conv1D(filters=640, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
#model.add(keras.layers.Dropout(0.002))
model.add(keras.layers.Conv1D(filters=320, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
#model.add(keras.layers.Dropout(0.002))
model.add(keras.layers.Conv1D(filters=160, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
#model.add(keras.layers.Dropout(0.002)) # Help prevent overfitting
model.add(keras.layers.Conv1D(filters=80, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
#model.add(keras.layers.Dropout(0.001))
model.add(keras.layers.Conv1D(filters=40, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
#model.add(keras.layers.Dropout(0.001))
model.add(keras.layers.Conv1D(filters=20, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
#model.add(keras.layers.Dropout(0.001))
model.add(keras.layers.Conv1D(filters=10, kernel_size=2, strides=1, padding="same"))
model.add(keras.layers.LeakyReLU(LRU))
model.add(keras.layers.Flatten())
### print layers
print(model.summary())

### training setting
opt = keras.optimizers.Adam(lr=0.00005)
str_metrics = ['mean_squared_error', 'mean_absolute_error']
str_labels = ['MSE', 'MAE']
loss = 'mse'


### train CNN
model.compile(optimizer=opt, loss=loss, metrics=str_metrics)

### early stop
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(input_train_norm, output_train_norm, epochs=1000,
                    validation_data = (input_val_norm, output_val_norm),
                    verbose=0, callbacks=[early_stop, PrintDot()])

# epochs: integer. Epochs is the number of times the model will cycle through the data
# the more epochs we run, the more the model will improve 
# verbose: how you want to see your output of your neural network while it's training
# verbose = 0(silent), 1- progress bar, 2- one line per epoch



### training history
tr_history = pd.DataFrame(history.history)
tr_history['epoch'] = history.epoch

## Make predictions
train_predictions_norm = model.predict(input_train_norm)
val_predictions_norm = model.predict(input_val_norm)
test_predictions_norm = model.predict(input_test_norm)
train_predictions = scaler_out.inverse_transform(train_predictions_norm)
val_predictions = scaler_out.inverse_transform(val_predictions_norm)
test_predictions = scaler_out.inverse_transform(test_predictions_norm)

## make output
cnn_data = {'input_train': input_train, 'input_test': input_test,'input_val': input_val,
            'output_train': output_train, 'output_test': output_test, 'output_val': output_val,
            'test_predictions':test_predictions,'val_predictions':val_predictions,'train_predictions':train_predictions,
            'str_metrics': str_metrics, 'str_labels': str_labels}

## save data
joblib.dump(scaler_in, 'scaler_in.pkl')
joblib.dump(scaler_out, 'scaler_out.pkl')
np.savez_compressed('cnn_data',**cnn_data)
model.save('cnn_model.csv')
with open('tr_history.pickle', 'wb') as f:
    pickle.dump(tr_history, f)

### convert to numpy array
M1_train = np.array(output_train)
M1_val = np.array(output_val)
M1_test = np.array(output_test)
M2_train = np.array(train_predictions)
M2_val = np.array(val_predictions)
M2_test = np.array(test_predictions)

M2 = np.concatenate((M2_train, M2_test, M2_val), axis=0)

## Writing the data 
'''
with open('predicting_model_2000.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(M2)
        f.close()
'''
### evaluations
### function to plot history
def plot_history(hist, str_metrics, str_labels):
    n = len(str_metrics)
    if n > 1:
        fig, ax = plt.subplots(n, 1)
        for ii in range(0, n):
            str_tr = str_metrics[ii]
            str_val = 'val_' + str_metrics[ii]
            str_y = str_labels[ii]
            ax[ii].plot(hist['epoch'], hist[str_tr], '-b', label='Training Error', linewidth=2)
            ax[ii].plot(hist['epoch'], hist[str_val], '-r', label='Validation Error', linewidth=2)
            ax[ii].grid()
            ax[ii].set_xlabel('Epoch')
            ax[ii].set_ylabel(str_y)
            ax[ii].legend()
    else:
        fig = plt.figure()
        ax = plt.gca()
        str_tr = str_metrics[0]
        str_val = 'val_' + str_metrics[0]
        str_y = str_labels[0]

        ax.plot(hist['epoch'], hist[str_tr], '-b', label='Training Error', linewidth=2)
        ax.plot(hist['epoch'], hist[str_val], '-r', label='Validation Error', linewidth=2)
        ax.grid()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(str_y)
        ax.legend()
    plt.show()



### function to plot error histogram
def plot_errorhist(nbins, M1_train, M2_train, M1_val, M2_val):
    error_train = M1_train.ravel() - M2_train.ravel() 
    error_val = M1_val.ravel() - M2_val.ravel()  
    fig = plt.figure()
    ax1 = fig.add_subplot(211)  
    ax2 = fig.add_subplot(212)
    ax1.hist(error_train, bins = nbins, color='b')
    ax1.set_title('Training')
    ax1.set_ylabel('Frequency')
    ax2.hist(error_val, bins = nbins, color='b')
    ax2.set_title('Validation')
    ax2.set_ylabel('Frequency')
    plt.show()

#Use for loop then plot them together
### plot history
plot_history(tr_history, str_metrics, str_labels)

### plot error residuals
plot_errorhist(50, M1_train, M2_train, M1_val, M2_val)

### function to plot the conductivity versus depth
depth = [-5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90, -95, -100]

plt.plot(M2_train[10][:], depth, label = "Predictive model")
plt.plot(M1_train[10][:], depth, label = "Actual model")
plt.xlabel('Modelings')
plt.ylabel('Depth')
plt.title('Modeling vs. Depth')
plt.legend()
plt.show()

    
