import json
import numpy as np
import random
from scipy import stats as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump, load
import h5py
import os
import math
from copy import deepcopy
from time import time
from tqdm.std import tqdm

RANDOM_SEED = 999
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class dataLoader(object):
    def __init__(self, path: str = '../kaggle_valued_shoppers/temp_data_cat.h5'):
        record = dict()
        self.h5f = h5py.File(path,'r')
        self.columns = ['transaction_recency','transaction_frequency','avg_past_transaction_value',
                        'offer_recency','offer_frequency','offer_occurred_flag','offer_goods_quantity','offer_value',
                        'purchased_goods_quantity','purchased_item_price']
        self.cols_X = [0,1,2,3,4,5,6,7]
        self.cols_Y = 8
        self.cols_Z = 9

    def read_data(self):
        data = self.h5f['temp_data'][:]
        self.h5f.close()
        return data

    def split_data(self, data):
        rand_ind = np.random.permutation(data.shape[0])
        self.train_customers = rand_ind[:162000]
        self.val_customers = rand_ind[162000:216000]
        self.test_customers = rand_ind[216000:]
        train_data = data[self.train_customers].T
        val_data = data[self.val_customers].T
        test_data = data[self.test_customers].T
        return train_data, val_data, test_data

    def split_columns(self, dataSplit):
        x = dataSplit[self.cols_X].T
        y = dataSplit[self.cols_Y].T
        z = dataSplit[self.cols_Z].T
        return x, y, z

    def unique_counts(self, y):
        unique, counts = np.unique(y[y!=0], return_counts=True)
        return np.asarray((unique, counts)).T
    
    def reshape(self,x, y):
        # cust * period, cat * variables
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2]*x.shape[3])
        # cust * period, cat * 1 (qty)
        y = y.reshape(y.shape[0]*y.shape[1],y.shape[2])
        return x, y

    def run(self):
        data = self.read_data()
        train_data, val_data, test_data = self.split_data(data)
        x_train, y_train, z_train = self.split_columns(train_data)
        x_val, y_val, z_val = self.split_columns(val_data)
        x_test, y_test, z_test = self.split_columns(test_data)
        print(self.unique_counts(y_test))
        x_train, y_train = self.reshape(x_train, y_train)
        x_test, y_test = self.reshape(x_test, y_test)
        return x_train, y_train, x_test, y_test


class train_rf(object):
    def __init__(self, subsample: int = 500000):
        self.subsample_size = subsample
        self.record = {}

    def subsample(self, x_train, subsample_size):
        # subsampling to make time of training the random forest manageable
        return np.random.choice(x_train.shape[0],subsample_size,replace=False)

    def subsample_same_data(self, data, n):
        index1 = np.random.choice(data.shape[0], n, replace=True)
        index2 = np.random.choice(data.shape[0], n, replace=True)
        sample1 = data[index1]
        sample2 = data[index2]
        return sample1, sample2

    def train_rf(self, x_train, y_train):
        self.clf = RandomForestRegressor(n_estimators=100, verbose=4, n_jobs=-1)
        self.clf = self.clf.fit(x_train, y_train)
        
    def train(self, x_train, y_train, model: str = 'rf'):
        index = self.subsample(x_train, self.subsample_size)
        if model == 'rf':
            self.train_rf(x_train[index], y_train[index])

    def get_bootstrap_kl(self, data_true, data_predicted, n_bins, x_range):
        n = data_true.shape[0]
        hist_true, _ = np.histogram(data_true, bins=n_bins, range=x_range)
        hist_predicted, _ = np.histogram(data_predicted, bins=n_bins, range=x_range)
        return sc.entropy(hist_true+1,hist_predicted+1), n

    def get_pval(self, simulated_KL, subsampled_KL, n_samples):
        subsampled_KL = sorted(subsampled_KL)
        pval = sum( simulated_KL < i for i in subsampled_KL) / float(n_samples)
        return pval
    
    def get_conf_interval(self, n_samples, subsampled_KL):
        samples = int(math.ceil(n_samples*0.95))
        conf_interval = (0,subsampled_KL[samples-1])
        return conf_interval
    
    # PERCENTILE KL DIVERGENCE BOOTSTRAP TEST
    def KL_validate(self, data_true, data_predicted, n_bins, x_range, n_samples=10000):
        '''"Pr(KL(simulated data||original) > KL(bootstrap original||bootstrap original))'''
        simulated_KL, n = self.get_bootstrap_kl(data_true, data_predicted, n_bins, x_range)
        subsampled_KL = []
        for _ in tqdm(range(n_samples), desc='Validating KL'):
            sample1, sample2 = self.subsample_same_data(data_true, n)
            kl, _ = self.get_bootstrap_kl(sample2, sample1, n_bins, x_range)
            subsampled_KL.append(kl)
        pval = self.get_pval(simulated_KL, subsampled_KL, n_samples)
        conf_interval = self.get_conf_interval(n_samples, subsampled_KL)
        return simulated_KL,conf_interval,pval,n

    def loss_functions(self, y_test, y_pred):
        self.record['MSE_rf'] = str(np.mean((y_test - y_pred)**2))
    
    def filter_predictions(self, y_pred, y_test):
        y_test_new = y_test[y_test!=0]
        y_pred_new = np.rint(y_pred)[(y_test!=0)&(np.rint(y_pred)!=0)].squeeze()
        return y_test_new, y_pred_new
    
    def sample_predictions(self, y_test_new, y_pred_new):
        index = self.subsample(y_test_new, 1000)
        y_test_new = y_test_new[index]
        index = self.subsample(y_pred_new, 1000)
        y_pred_new = y_pred_new[index]
        return y_test_new, y_pred_new

    def validate(self, x_test, y_test):
        y_pred = self.clf.predict(x_test)
        self.loss_functions(y_test, y_pred)
        y_test_new, y_pred_new = self.filter_predictions(y_pred, y_test)
        y_test_new, y_pred_new = self.sample_predictions(y_test_new, y_pred_new)
        self.record['KL_divergence_rf'] = str(self.KL_validate(y_test_new, y_pred_new, n_bins=4, x_range=(0,12)))

    def save_json(self, obj, name):
        with open(name, 'w') as outfile:
            json.dump(obj, outfile)

    def save_record(self, path = '../results/vs_record_regressor.json'):
        self.save_json(self.record, path)
        print(self.record)
    
    def save_model(self, path: str):
        dump(self.clf, path) 
    
    def run(self):
        dl = dataLoader('data/compressed/temp_data_cat.h5')
        x_train, y_train, x_test, y_test = dl.run()
        self.train(x_train, y_train, 'rf')
        self.validate(x_test, y_test)
        self.save_record('data/compressed/vs_record_regressor.json')
        self.save_model(f'rf_{time()}')

if __name__=='__main__':
    preprocess = train_rf()
    preprocess.run()