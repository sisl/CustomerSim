
# TRAINING KDD1998 REGRESSOR

from shared_functions import *
from net_designs import *

import pandas as ps
import numpy as np
from scipy import stats as sc
import random

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from keras.callbacks import ModelCheckpoint

# CREATE DICTIONARY TO HOLD RESULTS
record = dict()

# seed
RANDOM_SEED = 777
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# cupy.random.seed(RANDOM_SEED)

# LOAD DATA
print('Loading data')
data = ps.read_csv("../kdd98_data/kdd1998tuples.csv", header=None)
data.columns = ['customer','period','r0','f0','m0','ir0','if0','gender','age','income',
                'zip_region','zip_la','zip_lo','a','rew','r1','f1','m1','ir1','if1',
                'gender1','age1','income1','zip_region1','zip_la1','zip_lo1']
data['rew_ind'] = (data['rew'] > 0) * 1
data['age'][data['age'] == 0] = None

# PREPARE DATA
print('Preprocessing data')

# split into train, val, test
customers = list(set(data['customer']))

train_samples = 100000
val_samples = 50000
test_samples = len(customers) - val_samples - train_samples

np.random.shuffle(customers)

train_customers = customers[0:train_samples]
val_customers = customers[train_samples:(train_samples+val_samples)]
test_customers = customers[(train_samples+val_samples):]

cols = ['r0','f0','m0','ir0','if0','gender','age','income','zip_region','a','rew','rew_ind']

train_data = data[data['customer'].isin(train_customers) & data['rew_ind']==1][cols].fillna(0)
val_data = data[data['customer'].isin(val_customers) & data['rew_ind']==1][cols].fillna(0)
test_data = data[data['customer'].isin(test_customers) & data['rew_ind']==1][cols].fillna(0).sample(1000, random_state=RANDOM_SEED)

n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_test = test_data.shape[0]

cols_X = ['r0','f0','m0','ir0','if0','gender','age','income','zip_region','a']
cols_Y = ['rew']

x_train = train_data[cols_X].values.astype(np.float32)
y_train = train_data[cols_Y].values.astype(np.float32)

x_val = val_data[cols_X].values.astype(np.float32)
y_val = val_data[cols_Y].values.astype(np.float32)

x_test = test_data[cols_X].values.astype(np.float32)
y_test = test_data[cols_Y].values.astype(np.float32)


# DEFINE NEURAL NET
print('Training KDD98 regressor neural net')

n_epochs = 50
batch_size = 100
file_name="../results/kdd98_propagation_regressor_best.h5"

model = KDDRegressor()

# TRAIN NEURAL NET
model.compile(loss='mean_absolute_error', optimizer='adam')

# Callback to save the best model
checkpoint = ModelCheckpoint(file_name, monitor='val_loss', save_best_only=True, save_weights_only=True)

# Fit the model
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epochs,
    verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val))

# model.load_weights(file_name)

# model.save_weights(file_name, overwrite=True)
score = model.evaluate(x_test, y_test, verbose=1)
print('Test Loss: '+ str(score) )


# VALIDATE NEURAL NET
print('Validating neural net')

y_pred = model.predict(x_test)

record['test_mean'] = str(y_test.mean())
record['test_std'] = str(np.std(y_test))

record['KL_divergence_deeplearning'] = str(KL_validate(y_test.squeeze(), y_pred.squeeze(), n_bins=5, x_range=(0,50)))
record['prediction_mean_deeplearning'] = str(y_pred.mean())
record['prediction_std_deeplearning'] = str(np.std(y_pred))
record['MSE_deeplearning'] = str(np.mean((y_pred - y_test)**2))

plot_validate(y_test.squeeze(), y_pred.squeeze(), xlab="Donation Amount", ylab="Probability Mass", name="../results/kdd98_regressor.pdf", 
              n_bins=10, x_range=(0,50), y_range=(0,0.5), font = 20, legend=False, bar_width=1)


# TRAIN RANDOM FOREST
print('Training random forest')
clf = RandomForestRegressor(n_estimators=100)
clf = clf.fit(x_train, y_train.ravel())

# VALIDATE RANDOM FOREST
print('Validating random forest')
y_pred_rf = clf.predict(x_test)

record['KL_divergence_rf'] = str(KL_validate(y_test.squeeze(), y_pred_rf.squeeze(), n_bins=5, x_range=(0,50)))
record['prediction_mean_rf'] = str(y_pred_rf.mean())
record['prediction_std_rf'] = str(np.std(y_pred_rf))
record['MSE_rf'] = str(np.mean((y_pred_rf - y_test.ravel())**2))

plot_validate(y_test, y_pred_rf, xlab="Donation Amount ($)", ylab="Probability Mass", name="../results/kdd98_regressor_rf.pdf", 
              n_bins=10, x_range=(0,50), y_range=(0,0.5))

# SAVE RECORD
save_json(record,'../results/kdd98_record_regressor.json')
print(record)
