
# SIMULATE KDD1998

import sys
sys.path.insert(0, './src')

from shared_functions import *
from net_designs import *

import os
from scipy import stats as sc
import pandas as ps
import numpy as np
import random

from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner
from libpgm.tablecpdfactorization import TableCPDFactorization

import h5py

def sample_granular(net, bins, n_samples):
    sampled_data = net.randomsample(n=n_samples)
    granularize(sampled_data,bins)
    return ps.DataFrame(sampled_data)

def granularize(sampled_data,bins):
    for j in bins:
        na_level = len(bins[j])
        for i in xrange(len(sampled_data)):
            ind = sampled_data[i][j]
            if ind == na_level - 1:
                sampled_data[i][j] = None
            else:
                sampled_data[i][j] = round((np.random.uniform() * (bins[j][ind+1] - bins[j][ind]) + 
                                  bins[j][ind]))

def random_policy(data):
    return np.random.randint(0,12,data.shape[0])

def binom(x):
    return np.random.binomial(8,x)/8.0

def propagate(data, classifier, regressor, policy, threshold=0.275, periods=12, orig_actions=None):
    
    # Initializing arrays to hold output
    customers = np.zeros((periods+1,data.shape[0],data.shape[1]), dtype = np.float32)
    customers[0] = data
    actions = np.zeros((periods,data.shape[0]), dtype = np.float32)
    donations = np.zeros((periods,data.shape[0]), dtype = np.float32)
    
    for t in xrange(periods):
        
        # SELECTING ACTIONS
        if isinstance(orig_actions, (np.ndarray)):
            actions[t] = orig_actions[t]
        else:
            actions[t] = policy(customers[t])
        
        inp = np.append(customers[t],actions[t].reshape((data.shape[0],1)),axis = 1).astype(np.float32)
        
        # PROPAGATING CUSTOMERS
        donation_prob = classifier.predict_proba(inp,verbose=0)[:,1]
        
        donations_occurred = 1*(np.random.binomial(8,threshold,donation_prob.shape[0])/8.0 < np.apply_along_axis(binom, 0, donation_prob))
        donations[t] = np.rint(regressor.predict(inp,verbose=0).squeeze() * donations_occurred).astype(np.float32)
        
        # UPDATING CUSTOMER STATE
        
        # Recency
        customers[t+1,:,0] = (customers[t,:,0] + 1)*(donations_occurred == 0)
        
        # Frequency
        customers[t+1,:,1] = customers[t,:,1] + donations_occurred
        
        # Avg. Past Donation
        customers[t+1,:,2] = (customers[t,:,2] * customers[t,:,1] + donations[t]) / (customers[t+1,:,1] + 1*(customers[t+1,:,1]==0))
        
        # Avg. Interaction Recency
        customers[t+1,:,3] = (customers[t,:,3] + 1)*(actions[t] == 0) # Null action 0
        
        # Avg. Interaction Frequency
        customers[t+1,:,4] = customers[t,:,4] + (actions[t] != 0)

        customers[t+1,:,5:] = customers[t,:,5:]
    
    return customers, actions, donations


# LOAD MODELS
print('Loading models')

net_start_life = load("./results/kdd98_init_snapshot_start.p")
start_bins = load("./results/kdd98_init_snapshot_start_bins.p")
regressor = KDDRegressor()
regressor.load_weights("./results/kdd98_propagation_regressor_best.h5")
classifier = KDDClassifier()
classifier.load_weights("./results/kdd98_propagation_classifier_best.h5")

RANDOM_SEED = 999

# SIMULATION
test_samples = 1000
np.random.seed(RANDOM_SEED)

# SIMULATE INITIAL SNAPSHOT
sampled_data = sample_granular(net=net_start_life, bins=start_bins, n_samples=test_samples)
sampled_data = sampled_data.fillna(0)
sampled_data = sampled_data[['r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income', 'zip_region']].values

# SIMULATE THROUGH TIME - STATES, ACTIONS, DONATIONS
S, A, D = propagate(sampled_data, classifier, regressor, random_policy, threshold=0.275, periods=18)

# SAVE DATA
print('Saving data')
h5f = h5py.File('./results/kdd98_simulation_results.h5', 'w')
h5f.create_dataset('S', data=S)
h5f.create_dataset('A', data=A)
h5f.create_dataset('D', data=D)
h5f.close()

# LOAD DATA
#h5f = h5py.File('./results/kdd98_simulation_results.h5','r')
#S = h5f['S'][:]
#A = h5f['A'][:]
#D = h5f['D'][:]
#h5f.close()


