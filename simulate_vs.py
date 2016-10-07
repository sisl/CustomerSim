
# VS SIMULATE

import sys
sys.path.insert(0, './src')

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
    output = 1*(np.random.uniform(0,1,(data.shape[0],data.shape[1],1))<0.05)
    output = np.concatenate((output,output,5*output),axis=2)
    return output

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
        quantity_pred = np.rint(regressor.predict(inp,verbose=0)).astype(np.float32)
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


# LOAD MODEL
regressor = VSRegressor()
regressor.load_weights("./results/vs_propagation_quantity_best_cat.h5")

# LOAD AVG PRICES PER CATEGORY
print('Loading avg. prices')
prices = load('./kaggle_valued_shoppers/vs_cat_avg_prices.p')

RANDOM_SEED = 999

# SIMULATE DATA
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

initial_set = np.zeros((1000,20,5),dtype=np.float32)

# INPUT IS THE STARTING STATE ARRAY, NET, TESTED POLICY [IGNORED BECUASE WE PROVIDE ACTIONS FOR EVERY TIME PERIOD AS ORIG ACTIONS]
# AVG. PRICES FOR EACH CATEGORY, PERIODS FOR SMULATION, NUMBER OF VARIABLES DESCRIBING THE ACTION, [OPTIONAL] TENSOR OF ORIGINAL ACTIONS 
S, A, Q, P = propagate(initial_set, regressor, random_policy, prices, periods=16, num_actions=3)


# SAVE RECORD
# SAVE DATA
print('Saving data')
h5f = h5py.File('./results/vs_simulation_results.h5', 'w')
h5f.create_dataset('S', data=S)
h5f.create_dataset('A', data=A)
h5f.create_dataset('Q', data=Q)
h5f.create_dataset('P', data=P)
h5f.close()

# LOAD DATA
#h5f = h5py.File('./results/vs_simulation_results.h5','r')
#S = h5f['S'][:]
#A = h5f['A'][:]
#Q = h5f['Q'][:]
#P = h5f['P'][:]
#h5f.close()


