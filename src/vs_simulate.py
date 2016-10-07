
# VS SIMULATE

from shared_functions import *
from net_designs import *

import pandas as ps
import json
import numpy as np
import pickle
import networkx as nx

import scipy.stats as sct

from scipy import stats as sc
import random

import h5py
import os

from copy import deepcopy

from scipy import stats as sc



def random_policy(data):
    return np.random.randint(1,13,data.shape[0])

def propagate(data, regressor, policy, prices, periods=12, num_actions=3, orig_actions=None):
    
    # Initializing arrays to hold output
    customers = np.zeros((periods+1,data.shape[0],data.shape[1],data.shape[2]), dtype = np.float32)
    customers[0] = data
    
    actions = np.zeros((periods,data.shape[0],data.shape[1],num_actions), dtype = np.float32)
    quantity = np.zeros((periods,data.shape[0],data.shape[1]), dtype = np.float32)
    amount = np.zeros((periods,data.shape[0],data.shape[1]), dtype = np.float32)
    
    for t in xrange(periods):
        
        # SELECTING ACTIONS - IF A TENSOR OF ORIGINAL ACTIONS IS PROVIDED, IGNORE POLICY
        if isinstance(orig_actions, (np.ndarray)):
            actions[t] = orig_actions[t]
        # OTHERWISE, USE POLICY
        else:
            actions[t] = policy(customers[t])
        
        inp = np.append(customers[t],actions[t],axis = 2).astype(np.float32)
        inp = inp.reshape(inp.shape[0],-1)
        
        # PROPAGATING CUSTOMERS
        quantity_pred = np.rint(regressor.predict(inp)).astype(np.float32)
        quantity_pred[quantity_pred < 0] = 0
        quantity_sim = np.random.poisson(quantity_pred,quantity_pred.shape)
        quantity[t] = quantity_sim
        amount[t] = quantity[t] * np.tile(prices, (quantity[t].shape[0],1))
        
        # UPDATING CUSTOMER STATE
        
        # Recency
        customers[t+1,:,:,0] = (customers[t,:,:,0] + 1)*(quantity[t] == 0)
        
        # Frequency
        customers[t+1,:,:,1] = customers[t,:,:,1] + quantity[t]
        
        # Avg. Past Donation
        customers[t+1,:,:,2] = (customers[t,:,:,2] * customers[t,:,:,1] + amount[t]) / (customers[t+1,:,:,1] + 1*(customers[t+1,:,:,1]==0))
        
        # Offer Recency
        customers[t+1,:,:,3] = (customers[t,:,:,3] + 1)*(actions[t,:,:,0] == 0) # Null action 1
        
        # Offer Frequency
        customers[t+1,:,:,4] = customers[t,:,:,4] + actions[t,:,:,0]
    
    return customers, actions, quantity, amount


record = dict()

# LOAD MODEL
print('Loading model')
regressor = VSRegressor()
regressor.load_weights("../results/vs_propagation_quantity_best_cat.h5")

RANDOM_SEED = 999

# LOAD DATA
print('Loading data')

h5f = h5py.File('../kaggle_valued_shoppers/temp_data_cat.h5','r')
data = h5f['temp_data'][:]
h5f.close()

columns = ['transaction_recency','transaction_frequency','avg_past_transaction_value',
           'offer_recency','offer_frequency','number_of_offers','offer_goods_quantity_per_offer','offer_value_per_offer',
           'purchased_goods_quantity','purchased_item_price']

cols_X = [0,1,2,3,4,5,6,7]
cols_S = [0,1,2,3,4]
cols_A = [5,6,7]
cols_Y = 8
cols_Z = 9

# LOAD AVG PRICES PER CATEGORY
print('Loading avg. prices')
prices = load('../kaggle_valued_shoppers/vs_cat_avg_prices.p')

# PREPARE TEST DATA
print('Preparing data')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

rand_ind = np.random.permutation(data.shape[0])
test_customers = rand_ind[216000:217000] # 1000 customers from test set
test_data = data[test_customers].T

del data

# EXTRACT ARRAYS WITH ORIGINAL DATA

orig_S = test_data[cols_S].T
orig_A = test_data[cols_A].T
orig_Q = test_data[cols_Y].T
orig_P = test_data[cols_Z].T * orig_Q

del test_data

orig_S = np.transpose(orig_S, (1, 0, 2, 3))
orig_A = np.transpose(orig_A, (1, 0, 2, 3))
orig_Q = np.transpose(orig_Q, (1, 0, 2))
orig_P = np.transpose(orig_P, (1, 0, 2))


# SIMULATE DATA

T = 17

# INPUT IS THE STARTING STATE ARRAY, NET, TESTED POLICY [IGNORED BECUASE WE PROVIDE ACTIONS FOR EVERY TIME PERIOD AS ORIG ACTIONS]
# AVG. PRICES FOR EACH CATEGORY, PERIODS FOR SMULATION, NUMBER OF VARIABLES DESCRIBING THE ACTION, [OPTIONAL] TENSOR OF ORIGINAL ACTIONS 
S, A, Q, P = propagate(orig_S[0], regressor, random_policy, prices, 
                    periods=16, num_actions=3, orig_actions=orig_A)


# PLOT CUMULATIVE PURCHASES BY CATEGORY OVER TIME
plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
for i in range(Q.shape[2]):
    plt.plot(range(orig_Q.shape[0]),np.cumsum(orig_P.mean(1),0)[:,i], linewidth=2, alpha=0.3, color="green")
    plt.plot(range(Q.shape[0]),np.cumsum(P.mean(1),0)[:,i], linewidth=2, alpha=0.5, color="brown",linestyle='--')

plt.xlim(0,15)
plt.ylim(0,50)
line_green, = plt.plot([],[], label='Actual Data', color="green")
line_brown, = plt.plot([],[], label='Simulated Data', color="brown",linestyle='--')
#plt.legend(handles=[line_green, line_brown], fontsize=20)
plt.xlabel("Campaign Period", fontsize=20, labelpad=15)
plt.ylabel("Mean Cumulative Purchases", fontsize=20, labelpad=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.savefig("../results/vs_cumulative_purchase.pdf", bbox_inches='tight')
plt.close()


# PLOT HISTTOGRAM OF TOTAL PURCHASES
orig_purchases = np.sum(orig_P.sum(0),1).squeeze()
sim_purchases = np.sum(P.sum(0),1).squeeze()

plot_validate(orig_purchases, sim_purchases, xlab="Total Purchase Amount", ylab="Probability Mass", name="../results/vs_total_purchase.pdf", 
              n_bins=7, x_range=(0,525), y_range=(0,0.5), font = 20, legend=True, bar_width=15)


# MAKE A RECORD OF KEY PURCHASE METRICS
record['KL_divergence_deeplearning_purchases'] = str(KL_validate(orig_purchases, sim_purchases, 
    n_bins=7, x_range=(0,525)))
record['orig_mean_deeplearning_purchases'] = str(np.mean(orig_purchases))
record['sim_mean_deeplearning_purchases'] = str(np.mean(sim_purchases))
record['orig_std_deeplearning_purchases'] = str(np.std(orig_purchases))
record['sim_std_deeplearning_purchases'] = str(np.std(sim_purchases))


# CALCULATE THE NUMBER OF SIGNIFICANTLY DIFFERENT SIMULATED TOTAL PURCHASE HISOTGRAMS - BY CATEGORY
g = 0
for i in range(20):
    a = KL_validate(orig_P.sum(0)[:,i].squeeze(), P.sum(0)[:,i].squeeze(), n_bins=7, x_range=(0,525))
    record['KL_purchases_'+str(i)] = str(a)
    g+= 1*(a[2]<0.05)

record['KL_purchases_purchases_significant'] = g 


# PLOT MEAN RECENCY OVER TIME
plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
for i in range(S.shape[2]):
    plt.plot(range(orig_S.shape[0]),orig_S.mean(1)[:,i,0], linewidth=2, alpha=0.3, color="green")
    plt.plot(range(S.shape[0]),S.mean(1)[:,i,0], linewidth=2, alpha=0.5, color="brown",linestyle='--')

plt.xlim(0,15)
plt.ylim(0,20)
line_green, = plt.plot([],[], label='Actual Data', color="green")
line_brown, = plt.plot([],[], label='Simulated Data', color="brown",linestyle='--')
plt.legend(handles=[line_green, line_brown], fontsize=20)
plt.xlabel("Campaign Period", fontsize=20, labelpad=15)
plt.ylabel("Mean Transaction Recency", fontsize=20, labelpad=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.savefig("../results/vs_recency.pdf", bbox_inches='tight')
plt.close()

# PLOT HISTOGRAM OF ENDPERIOD RECENCY

orig_recen = np.mean(orig_S[16],1)[:,0].squeeze()
sim_recen = np.mean(S[16],1)[:,0].squeeze()

plot_validate(orig_recen,sim_recen, xlab="End-Period Recency", ylab="Probability Mass", name="../results/vs_endperiod_recency.pdf", 
              n_bins=5, x_range=(0,20), y_range=(0,0.5), font = 20, legend=True, bar_width=1)

# MAKE A RECORD OF KEY RECENCY METRICS
record['KL_divergence_deeplearning_recen'] = str(KL_validate(orig_recen, sim_recen, n_bins=5, x_range=(0,20)))
record['orig_mean_deeplearning_recen'] = str(np.mean(orig_recen))
record['sim_mean_deeplearning_recen'] = str(np.mean(sim_recen))
record['orig_std_deeplearning_recen'] = str(np.std(orig_recen))
record['sim_std_deeplearning_recen'] = str(np.std(sim_recen))


# SAVE RECORD
save_json(record,'../results/vs_record_simulate.json')
print(record)

