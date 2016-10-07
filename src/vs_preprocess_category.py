# PREPROCESS VALUED SHOPPERS DATA

# https://www.kaggle.com/c/acquire-valued-shoppers-challenge/forums/t/7666/getting-started-data-reduction

from shared_functions import *

import pandas as ps
import json
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import h5py

from datetime import datetime

# EXTRACTING DATA

print('Extracting VS data')

folder = '../kaggle_valued_shoppers/'

loc_offers = folder + "offers"
loc_train_history = folder + 'trainHistory'
loc_test_history = folder + "testHistory"
loc_transactions = folder + "transactions"
loc_reduced = folder + "reduced_transactions.csv"

def reduce_data(loc_offers, loc_transactions, loc_reduced):

    start = datetime.now()
    #get all categories on offer in a dict
    offers = {}
    for e, line in enumerate( open(loc_offers) ):
        offers[ line.split(",")[1] ] = 1
    #open output file
    with open(loc_reduced, "wb") as outfile:
        #go through transactions file and reduce
        reduced = 0
        for e, line in enumerate( open(loc_transactions) ):
            if e == 0:
                outfile.write( line ) #print header
            else:
                #only write when category in offers dict
                if line.split(",")[3] in offers:
                    outfile.write( line )
                    reduced += 1
            #progress
            if e % 5000000 == 0:
                print e, reduced, datetime.now() - start
    print e, reduced, datetime.now() - start


reduce_data(loc_offers, loc_transactions, loc_reduced)

# LOADING AND PROCESSING DATA
print('Loading and processing VS data')

def date_to_int(date_string):
    return int(date_string.replace("-", "")[:6])

offers = ps.read_csv(loc_offers)
train_history = ps.read_csv(loc_train_history)
test_history = ps.read_csv(loc_test_history)
history = ps.concat([train_history[test_history.columns],test_history])
trns = ps.read_csv(loc_reduced)

trns['period'] = trns['date'].apply(date_to_int)
history['period'] = history['offerdate'].apply(date_to_int)

trns['cacobr'] = trns['category'].apply(str)
offers['cacobr'] = offers['category'].apply(str)

full_offer_history = history.merge(offers, on="offer", how='left')
full_offer_history['offer_ind'] = 1

# dropping returns + strange transactions with 0 purchase quantity
trns = trns[trns['purchasequantity']>0]

customers = set(trns['id'])
num_customers = len(customers)
trns = trns[trns['id'].isin(customers)]

trns_cacobr = set(trns['cacobr'])
offers_cacobr = set(offers['cacobr'])
trns_period = set(trns['period'])
history_period = set(full_offer_history['period'])

data = np.zeros((len(customers),len(trns_period),len(trns_cacobr),10),dtype=np.float32)

ordered_customers = sorted(list(customers))
ordered_periods = sorted(list(trns_period))
ordered_cacobr = sorted(list(trns_cacobr))

customer_index = dict(zip(ordered_customers, range(len(ordered_customers))))
period_index = dict(zip(ordered_periods, range(len(ordered_periods))))
cacobr_index = dict(zip(ordered_cacobr, range(len(ordered_cacobr))))

purchase_data = ps.pivot_table(trns, values=['purchasequantity','purchaseamount'], index=['id','period','cacobr'], aggfunc=np.sum)
offer_data = ps.pivot_table(full_offer_history, values=['offer_ind','quantity','offervalue'], index=['id','period','cacobr'], aggfunc=np.sum)

# RECORDING ALL OFFERS
print('Recording offer data')
for i in xrange(offer_data.shape[0]):
    temp = offer_data.iloc[i]
    if not temp.name[0] in customers:
        continue
    ind = (customer_index[temp.name[0]],
           period_index[temp.name[1]],
           cacobr_index[temp.name[2]])
    
    data[ind][5] = temp.values[0]
    data[ind][6] = temp.values[2] / temp.values[0]
    data[ind][7] = temp.values[1] / temp.values[0]
    if i % 50000 == 0:
        print i

# RECORDING ALL TRANSACTIONS
print('Recording transaction data')
for i in xrange(purchase_data.shape[0]):
    temp = purchase_data.iloc[i]
    ind = (customer_index[temp.name[0]],
           period_index[temp.name[1]],
           cacobr_index[temp.name[2]])
    
    data[ind][8] = temp.values[1]
    data[ind][9] = temp.values[0] / temp.values[1]
    if i % 200000 == 0:
        print i

# CALCULATING RFM-I METRICS
print('Calculating RFM-I metrics')
for p in xrange(1,len(trns_period)):
    
    # update transaction RFM
    data[:,p,:,0] = (data[:,p-1,:,0] + 1) * (data[:,p-1,:,8] == 0)
    data[:,p,:,1] = data[:,p-1,:,1] + data[:,p-1,:,8]
    data[:,p,:,2] = (data[:,p-1,:,1] * data[:,p-1,:,2] + data[:,p-1,:,8] * data[:,p-1,:,9]) / (data[:,p,:,1] + 1*(data[:,p,:,1] == 0))
    
    # update offer RF
    data[:,p,:,3] = (data[:,p-1,:,3] + 1) * (data[:,p-1,:,5] == 0)
    data[:,p,:,4] = data[:,p-1,:,4] + data[:,p-1,:,5]
    
    print p

# SAVE DATA
print('Saving data')
h5f = h5py.File(folder+'temp_data_cat.h5', 'w')
h5f.create_dataset('temp_data', data=data)
h5f.close()

# LOAD DATA
#h5f = h5py.File(folder + 'temp_data_cat.h5','r')
#data = h5f['temp_data'][:]
#h5f.close()

# CALCULATE AVG PRICES PER CATEGORY
print('Calculating avg. prices')
prices = np.zeros(data.shape[2],dtype=np.float32)
for i in range(data.shape[2]):
    temp = data[:,:,i]
    prices[i] = temp[temp!=0].mean()

save(prices,folder+'vs_cat_avg_prices.p')


columns = ['transaction_recency','transaction_frequency','avg_past_transaction_value',
           'offer_recency','offer_frequency','offer_occurred_flag','offer_goods_quantity','offer_value',
           'goods_quantity','item_price']
