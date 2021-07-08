# PREPROCESS VALUED SHOPPERS DATA

# https://www.kaggle.com/c/acquire-valued-shoppers-challenge/forums/t/7666/getting-started-data-reduction

# from shared_functions import *

import pandas as pd
# import json
import numpy as np
import pickle
# import networkx as nx
# import matplotlib.pyplot as plt
import os
# import random
import h5py
from tqdm import tqdm
from datetime import datetime

# EXTRACTING DATA

class Preprocess(object):
    def __init__(self, folder: str = 'data/compressed'):
        print("Init preprocess")
        self.paths = self.get_paths(folder)

    def get_paths(self, folder):
        return {'loc_offers': os.path.join(folder, "offers.csv.gz"),
                'loc_train_history': os.path.join(folder, 'trainHistory.csv.gz'),
                'loc_test_history': os.path.join(folder, "testHistory.csv.gz"),
                'loc_transactions': os.path.join(folder, "transactions.csv.gz"),
                'loc_reduced': os.path.join(folder, "reduced_transactions.csv"),
                'temp_data_cat': os.path.join(folder, "temp_data_cat.h5")}

    def reduce_data(self):
        start = datetime.now()
        #get all categories on offer in a dict
        offers = pd.read_csv('data/compressed/offers.csv.gz', compression='gzip')
        #open output file
        chunksize=1000
        header = pd.read_csv(self.paths['loc_transactions'], compression='gzip', nrows=0)
        header.to_csv(self.paths['loc_reduced'], index=False)
        with pd.read_csv(self.paths['loc_transactions'], compression='gzip', chunksize=chunksize) as reader:
            for chunk in tqdm(reader):
                chunk = chunk[chunk['category'].isin(offers['category'].values)]
                chunk.to_csv(self.paths['loc_reduced'], index=False, header=False, mode='a')
        print(f"Finished reducing in: {datetime.now() - start}")

    def date_to_int(self, date_string):
        return int(date_string.replace("-", "")[:6])

    def load_datasets(self):
        offers = pd.read_csv(self.paths['loc_offers'], compression='gzip')
        trns = pd.read_csv(self.paths['loc_reduced'])
        train_history = pd.read_csv(self.paths['loc_train_history'], compression='gzip')
        test_history = pd.read_csv(self.paths['loc_test_history'], compression='gzip')
        history = pd.concat([train_history[test_history.columns],test_history])
        return offers, trns, history

    def format_data(self, trns, history, offers):
        trns['period'] = trns['date'].apply(self.date_to_int)
        trns['cacobr'] = trns['category'].apply(str)

        history['period'] = history['offerdate'].apply(self.date_to_int)
        offers['cacobr'] = offers['category'].apply(str)
        return trns, history, offers

    def make_full_history(self, history, offers):
        full_offer_history = history.merge(offers, on="offer", how='left')
        full_offer_history['offer_ind'] = 1
        return full_offer_history

    def sanity_clean(self, trns):
        # # dropping returns + strange transactions with 0 purchase quantity
        trns = trns[trns['purchasequantity']>0]
        customers = set(trns['id'])
        trns = trns[trns['id'].isin(customers)]
        return trns
    
    def vectorize(self, trns, offers, full_offer_history):
        customers = set(trns['id'])
        trns_cacobr = set(trns['cacobr'])
        trns_period = set(trns['period'])
        data = np.zeros((len(customers),len(trns_period),len(trns_cacobr),10), dtype=np.float32)
        customer_index, period_index, cacobr_index = self.create_indexes(customers, trns_period, trns_cacobr)
        return data, customer_index, period_index, cacobr_index, customers, trns_period
    
    def create_indexes(self, customers, trns_period, trns_cacobr):
        ordered_customers = sorted(list(customers))
        ordered_periods = sorted(list(trns_period))
        ordered_cacobr = sorted(list(trns_cacobr))
        customer_index = dict(zip(ordered_customers, range(len(ordered_customers))))
        period_index = dict(zip(ordered_periods, range(len(ordered_periods))))
        cacobr_index = dict(zip(ordered_cacobr, range(len(ordered_cacobr))))
        return customer_index, period_index, cacobr_index
    
    def pivot(self, trns, full_offer_history):
        purchase_data = pd.pivot_table(trns, values=['purchasequantity', 'purchaseamount'], index=['id','period','cacobr'], aggfunc=np.sum)
        offer_data = pd.pivot_table(full_offer_history, values=['offer_ind', 'quantity', 'offervalue'], index=['id','period','cacobr'], aggfunc=np.sum)
        return purchase_data, offer_data

    def record_offers(self, offer_data, customers, customer_index, period_index, cacobr_index, data):
        for ix, row in tqdm(offer_data.iterrows()):
            if not row.name[0] in customers:
                continue
            ind = (customer_index[row.name[0]],
                   period_index[row.name[1]],
                   cacobr_index[row.name[2]])
            data[ind][5] = row.values[0]
            data[ind][6] = row.values[2] / row.values[0]
            data[ind][7] = row.values[1] / row.values[0]
        return data
    
    def record_transactions(self, purchase_data, customer_index, period_index, cacobr_index, data):
        for ix, row in tqdm(purchase_data.iterrows()):
            ind = (customer_index[row.name[0]],
                   period_index[row.name[1]],
                   cacobr_index[row.name[2]])
            data[ind][8] = row.values[1]
            data[ind][9] = row.values[0] / row.values[1]
        return data
    
    def build_rfmi(self, trns_period, data):
        for p in tqdm(range(1,len(trns_period))):
            # update transaction RFM
            data[:,p,:,0] = (data[:,p-1,:,0] + 1) * (data[:,p-1,:,8] == 0)
            data[:,p,:,1] = data[:,p-1,:,1] + data[:,p-1,:,8]
            data[:,p,:,2] = (data[:,p-1,:,1] * data[:,p-1,:,2] + data[:,p-1,:,8] * data[:,p-1,:,9]) / (data[:,p,:,1] + 1*(data[:,p,:,1] == 0))
            
            # update offer RF
            data[:,p,:,3] = (data[:,p-1,:,3] + 1) * (data[:,p-1,:,5] == 0)
            data[:,p,:,4] = data[:,p-1,:,4] + data[:,p-1,:,5]
        return data
    
    def save_h5(self, data, path):
        h5f = h5py.File(path, 'w')
        h5f.create_dataset('temp_data', data=data)
        h5f.close()
    
    def save_pickle(self, obj, name):
        pickle.dump(obj, open(name, "wb"))
    
    def run(self):
        # self.reduce_data()
        offers, trns, history = self.load_datasets()
        trns, history, offers = self.format_data(trns, history, offers)
        full_offer_history = self.make_full_history(history, offers)
        trns = self.sanity_clean(trns)
        data, customer_index, period_index, cacobr_index, customers, trns_period = self.vectorize(trns, offers, full_offer_history)
        purchase_data, offer_data = self.pivot(trns, full_offer_history)
        data = self.record_offers(offer_data, customers, customer_index, period_index, cacobr_index, data)
        data = self.record_transactions(purchase_data, customer_index, period_index, cacobr_index, data)
        data = self.build_rfmi(trns_period, data)
        self.save_h5(data, self.paths['temp_data_cat'])
        prices = self.avg_price_per_category(data)
        self.save_pickle(prices,'data/compressed/vs_cat_avg_prices.p')
        
    def avg_price_per_category(self, data):
        prices = np.zeros(data.shape[2],dtype=np.float32)
        for i in range(data.shape[2]):
            temp = data[:,:,i]
            prices[i] = temp[temp!=0].mean()
        return prices

def load_data(file: str = 'data/compressed/temp_data_cat.h5'):
    h5f = h5py.File(file, 'r')
    data = h5f['temp_data'][:]
    h5f.close()
    return data

# columns = ['transaction_recency','transaction_frequency','avg_past_transaction_value',
#            'offer_recency','offer_frequency','offer_occurred_flag','offer_goods_quantity','offer_value',
#            'goods_quantity','item_price']

if __name__=='__main__':
    preprocess = Preprocess()
    preprocess.run()