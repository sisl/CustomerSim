
# VS REGRESSOR

from shared_functions import *
from net_designs import *

import pandas as ps
import json
import numpy as np
import pickle
import networkx as nx
import random

import scipy.stats as sct

from scipy import stats as sc

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import h5py
import os

from copy import deepcopy

RANDOM_SEED = 999

record = dict()

# LOAD DATA
print('Loading data')

h5f = h5py.File('../kaggle_valued_shoppers/temp_data_cat.h5','r')
data = h5f['temp_data'][:]
h5f.close()

columns = ['transaction_recency','transaction_frequency','avg_past_transaction_value',
           'offer_recency','offer_frequency','offer_occurred_flag','offer_goods_quantity','offer_value',
           'purchased_goods_quantity','purchased_item_price']

cols_X = [0,1,2,3,4,5,6,7]
cols_Y = 8
cols_Z = 9

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

rand_ind = np.random.permutation(data.shape[0])
train_customers = rand_ind[:162000]
val_customers = rand_ind[162000:216000]
test_customers = rand_ind[216000:]

train_data = data[train_customers].T
val_data = data[val_customers].T
test_data = data[test_customers].T

del data

train_data.shape

x_train = train_data[cols_X].T
y_train = train_data[cols_Y].T
z_train = train_data[cols_Z].T
del train_data

x_val = val_data[cols_X].T
y_val = val_data[cols_Y].T
z_val = val_data[cols_Z].T
del val_data

x_test = test_data[cols_X].T
y_test = test_data[cols_Y].T
z_test = test_data[cols_Z].T
del test_data

unique, counts = np.unique(y_train[y_train!=0], return_counts=True)
print np.asarray((unique, counts)).T


x_train_reshaped = x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2]*x_train.shape[3])
y_train_reshaped = y_train.reshape(y_train.shape[0]*y_train.shape[1],y_train.shape[2])

x_val_reshaped = x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2]*x_val.shape[3])
y_val_reshaped = y_val.reshape(y_val.shape[0]*y_val.shape[1],y_val.shape[2])

x_test_reshaped = x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2]*x_test.shape[3])
y_test_reshaped = y_test.reshape(y_test.shape[0]*y_test.shape[1],y_test.shape[2])


# TRAIN NEURAL NET
print('Training VS regressor neural net')

n_epochs = 10
batch_size = 500
file_name="../results/vs_propagation_quantity_best_cat.h5"

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model = VSRegressor()
model.compile(loss='mse', optimizer='adagrad')

# Callback to save the best model
checkpoint = ModelCheckpoint(file_name, monitor='val_loss', save_best_only=True, save_weights_only=True)

# Fit the model
model.fit(x_train_reshaped, y_train_reshaped, batch_size=batch_size, nb_epoch=n_epochs,
    verbose=1, callbacks=[checkpoint], validation_data=(x_val_reshaped, y_val_reshaped))

model.load_weights(file_name)

# model.save_weights(file_name, overwrite=True)
score = model.evaluate(x_test_reshaped, y_test_reshaped, verbose=1)
print('Test Loss: '+ str(score) )


# VALIDATE NEURAL NET

y_pred_reshaped = model.predict(x_test_reshaped)
record['MSE_deep_learning'] = str(np.mean((y_test_reshaped - y_pred_reshaped)**2))

y_test_dl = y_test_reshaped[y_test_reshaped!=0]
y_pred_dl = np.rint(y_pred_reshaped)[(y_test_reshaped!=0)&(np.rint(y_pred_reshaped)!=0)].squeeze()

index = np.random.choice(y_test_dl.shape[0],1000,replace=False)
y_test_dl = y_test_dl[index]

index = np.random.choice(y_pred_dl.shape[0],1000,replace=False)
y_pred_dl = y_pred_dl[index]

record['orig_mean_deep_learning'] = str(np.mean(y_test_dl))
record['sim_mean_deep_learning'] = str(np.mean(y_pred_dl))
record['orig_std_deep_learning'] = str(np.std(y_test_dl))
record['sim_std_deep_learning'] = str(np.std(y_pred_dl))

plot_validate(y_test_dl, y_pred_dl, xlab="Purchase Quantity", ylab="Probability Mass", name="../results/vs_regressor.pdf", 
              n_bins=12, x_range=(1,12), y_range=(0,0.8), font = 20, legend=True, bar_width=0.5)

record['KL_divergence_deeplearning'] = str(KL_validate(y_test_dl, y_pred_dl, n_bins=4, x_range=(0,12)))


# TRAIN RANDOM FOREST
print('Training random forest')
x_train_rf = x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2]*x_train.shape[3])
y_train_rf = y_train.reshape(y_train.shape[0]*y_train.shape[1],y_train.shape[2])

index = np.random.choice(x_train_rf.shape[0],500000,replace=False) # subsampling to make time of training the random forest manageable

clf = RandomForestRegressor(n_estimators=100)
clf = clf.fit(x_train_rf[index], y_train_rf[index])

# VALIDATE RANDOM FOREST

x_test_rf = x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2]*x_test.shape[3])
y_test_rf = y_test.reshape(y_test.shape[0]*y_test.shape[1],y_test.shape[2])

y_pred_rf = clf.predict(x_test_rf)

record['MSE_rf'] = str(np.mean((y_test_rf - y_pred_rf)**2))

y_test_rf_new = y_test_rf[y_test_rf!=0]
y_pred_rf_new = np.rint(y_pred_rf)[(y_test_rf!=0)&(np.rint(y_pred_rf)!=0)].squeeze()

index = np.random.choice(y_test_rf_new.shape[0],1000,replace=False)
y_test_rf_new = y_test_rf_new[index]

index = np.random.choice(y_pred_rf_new.shape[0],1000,replace=False)
y_pred_rf_new = y_pred_rf_new[index]

record['KL_divergence_rf'] = str(KL_validate(y_test_rf_new, y_pred_rf_new, n_bins=4, x_range=(0,12)))

# SAVE RECORD
save_json(record,'../results/vs_record_regressor.json')
print(record)
