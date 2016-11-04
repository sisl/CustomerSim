
# SIMULATE KDD1998

from shared_functions import *
from net_designs import *

import pandas as ps
import numpy as np
import networkx as nx
import random

import os

from copy import deepcopy
from scipy import stats as sc

from sklearn.metrics import roc_curve, auc

from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner
from libpgm.tablecpdfactorization import TableCPDFactorization


RANDOM_SEED = 777

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
        donation_prob = classifier.predict_proba(inp)[:,1]
        
        donations_occurred = 1*(np.random.binomial(8,threshold,donation_prob.shape[0])/8.0 < np.apply_along_axis(binom, 0, donation_prob))
        donations[t] = np.rint(regressor.predict(inp).squeeze() * donations_occurred).astype(np.float32)
        
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


# CREATE DICTIONARY TO HOLD RESULTS
record = dict()

# LOAD MODELS
print('Loading models')
net_start_life = load("../results/kdd98_init_snapshot_start.p")
start_bins = load("../results/kdd98_init_snapshot_start_bins.p")
regressor = KDDRegressor()
regressor.load_weights("../results/kdd98_propagation_regressor_best.h5")
classifier = KDDClassifier()
classifier.load_weights("../results/kdd98_propagation_classifier_best.h5")

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
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

customers = list(set(data['customer']))
train_samples = 100000
val_samples = 50000
test_samples = min(len(customers) - val_samples - train_samples,1000)

np.random.shuffle(customers)
train_customers = customers[0:train_samples]
val_customers = customers[train_samples:(train_samples+val_samples)]
test_customers = customers[(train_samples+val_samples):(train_samples+val_samples+test_samples)]
len(test_customers)


# SIMULATION
print('Simulating...')

# SIMULATE INITIAL SNAPSHOT - AS AN EXAMPLE - BELOW WE USE ACTUAL DATA AS A STARTING POINT FOR SIMULATION - FOR VALIDATION
sampled_data = sample_granular(net=net_start_life, bins=start_bins, n_samples=len(test_customers))
sampled_data = sampled_data.fillna(0)
sampled_data = sampled_data[['r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income', 'zip_region']].values

# EXTRACT ORIGINAL DATA FOR 22 PERIODS
data_orig = data[data['customer'].isin(test_customers)].fillna(0)

cols = ['r0','f0','m0','ir0','if0','gender','age','income','zip_region','a','rew','rew_ind']
cols_simX = ['r0','f0','m0','ir0','if0','gender','age','income','zip_region']
cols_simnX = ['r1','f1','m1','ir1','if1','gender','age','income','zip_region']
cols_simY = ['rew']

T = 22

orig_S = np.zeros((T+1,test_samples,9), dtype = np.float32)
orig_A = np.zeros((T,test_samples), dtype = np.float32)
orig_D = np.zeros((T,test_samples), dtype = np.float32)

temp = data_orig[data_orig['period']==1]
orig_S[0] = temp[cols_simX]
for t in range(22):
    temp = data_orig[data_orig['period']==(t+1)]
    orig_S[t+1] = temp[cols_simnX]
    orig_A[t] = temp['a']
    orig_D[t] = temp['rew']

# ISOLATE ORIGINAL DATA STARTING IN PERIOD 4
orig_S = orig_S[3:22]
orig_A = orig_A[3:21]
orig_D = orig_D[3:21]


# SIMULATE THROUGH TIME
S, A, D = propagate(orig_S[0], classifier, regressor, random_policy, 
    threshold=0.275, periods=18, orig_actions=orig_A)


print('Validating the simulator')

# PLOT CUMULATIVE DONATIONS OVER TIME
m = np.mean(np.cumsum(D,0),1)
s = np.std(np.cumsum(D,0),1)
m_orig = np.mean(np.cumsum(orig_D,0),1)
s_orig = np.std(np.cumsum(orig_D,0),1)

plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
plt.plot(range(D.shape[0]), m_orig, linewidth=2, alpha=1, color='green')
plt.plot(range(D.shape[0]), m_orig+s_orig, linewidth=2, alpha=0.8, color='green',linestyle='--')
plt.plot(range(D.shape[0]), m_orig+2*s_orig, linewidth=2, alpha=0.8, color='green',linestyle='--')
plt.plot(range(D.shape[0]), m, linewidth=2, alpha=1, color='brown')
plt.plot(range(D.shape[0]), m+s, linewidth=2, alpha=0.8, color='brown',linestyle='--')
plt.plot(range(D.shape[0]), m+2*s, linewidth=2, alpha=0.8, color='brown',linestyle='--')
plt.xlim(0,17)
plt.ylim(0,150)
line_green, = plt.plot([],[], label='Actual Data', color='green')
line_brown, = plt.plot([],[], label='Simulated Data', color='brown')
#plt.legend(handles=[line_green, line_brown],fontsize=20)
plt.xlabel("Campaign Period", fontsize=20, labelpad=15)
plt.ylabel("Mean Cumulative Donations", fontsize=20, labelpad=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.savefig("../results/kdd98_cumulative_donations.pdf", bbox_inches='tight')
plt.close()

# PLOT AVG. RECENCY OVER TIME
m_s = np.mean(S,1)
s_s = np.std(S,1)
m_s_orig = np.mean(orig_S,1)
s_s_orig = np.std(orig_S,1)

plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
plt.plot(range(S.shape[0]), m_s_orig[:,0], linewidth=2, alpha=1, color='green')
plt.plot(range(S.shape[0]), m_s_orig[:,0]+s_s_orig[:,0], linewidth=2, alpha=0.8, color='green',linestyle='--')
plt.plot(range(S.shape[0]), m_s_orig[:,0]+2*s_s_orig[:,0], linewidth=2, alpha=0.8, color='green',linestyle='--')
plt.plot(range(S.shape[0]), m_s[:,0], linewidth=2, alpha=1, color='brown')
plt.plot(range(S.shape[0]), m_s[:,0]+s_s[:,0], linewidth=2, alpha=0.8, color='brown',linestyle='--')
plt.plot(range(S.shape[0]), m_s[:,0]+2*s_s[:,0], linewidth=2, alpha=0.8, color='brown',linestyle='--')
plt.xlim(0,17)
plt.ylim(0,30)
line_green, = plt.plot([],[], label='Actual Data', color='green')
line_brown, = plt.plot([],[], label='Simulated Data', color='brown')
plt.legend(handles=[line_green, line_brown])
plt.xlabel("Campaign Period", fontsize=20, labelpad=15)
plt.ylabel("Mean Transaction Recency", fontsize=20, labelpad=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.legend(fontsize=20)
plt.savefig("../results/kdd98_recency.pdf", bbox_inches='tight')
plt.close()

# PLOT HISTOGRAM OF TOTAL DONATIONS
plot_validate(orig_D.sum(0),D.sum(0), xlab="Total Donations", ylab="Probability Mass", name="../results/kdd98_total_donations.pdf", 
              n_bins=5, x_range=(0,100), y_range=(0,0.6), font = 20, legend=True, bar_width=3)

record['KL_divergence_total_donations'] = str(KL_validate(orig_D.sum(0), D.sum(0), n_bins=5, x_range=(0,100)))
record['mean_total_donations_sim'] = str(np.mean(D.sum(0)))
record['mean_total_donations_orig'] = str(np.mean(orig_D.sum(0)))
record['std_total_donations_sim'] = str(np.std(D.sum(0)))
record['std_total_donations_orig'] = str(np.std(orig_D.sum(0)))

# PLOT HISTOGRAM OF END-PERIOD RECENCY
plot_validate(orig_S[18,:,0],S[18,:,0], xlab="End-Period Recency", ylab="Probability Mass", name="../results/kdd98_end_recency.pdf", 
              n_bins=5, x_range=(0,25), y_range=(0,0.8), font = 20, legend=True, bar_width=1)

record['KL_divergence_end_recency'] = str(KL_validate(orig_S[18,:,0], S[18,:,0], n_bins=5, x_range=(0,25)))
record['mean_end_recency_sim'] = str(np.mean(S[18,:,0]))
record['mean_end_recency_orig'] = str(np.mean(orig_S[18,:,0]))
record['std_end_recency_sim'] = str(np.std(S[18,:,0]))
record['std_end_recency_orig'] = str(np.std(orig_S[18,:,0]))

# SAVE RECORD
save_json(record,'../results/kdd98_record_simulation.json')
print(record)
