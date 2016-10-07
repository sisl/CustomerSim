
# KDD1998 - Generating initial customer snapshot

from shared_functions import *

import pandas as ps
ps.options.mode.chained_assignment = None

import numpy as np
import networkx as nx
import random

import os
from scipy import stats as sc
import math

from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner
from libpgm.tablecpdfactorization import TableCPDFactorization

RANDOM_SEED = 777

def discretize(data, vars_to_discretize, n_bins):
    
    '''
    Accepts data, a dictionary containing dicretization type for selected variables, and 
    a dictionary containing the number of bins for selected variables.

    Returns data after selected variables have been discretized, 
    together with binning definition for each variable.
    '''
    
    data_subset = ps.DataFrame(data).copy()
    bins = {}
    for i in vars_to_discretize:
        
        out = None
        binning = None
        
        # discretize by splitting into equal intervals
        if vars_to_discretize[i] == 'Equal': 
            out, binning = ps.cut(data_subset.ix[:,i],bins=n_bins[i],labels=False,retbins=True)

        # discretize by frequency
        elif vars_to_discretize[i] == 'Freq':
            nb = n_bins[i]
            while True:
                try:
                    out, binning = ps.qcut(data_subset.ix[:,i],q=nb,labels=False,retbins=True)
                    break
                except:
                    nb -= 1

        # discretize based on provided bin margins
        elif vars_to_discretize[i] == 'Bins':
            out = np.digitize(data_subset.ix[:,i], n_bins[i], right=True) - 1
            binning = n_bins[i]
                
        data_subset.ix[:,i] = out

        # replace NA variables with and special index (1+max) - 
        # if it has not been done so automatically an in np.digitize
        data_subset.ix[:,i][data_subset.ix[:,i].isnull()] = data_subset.ix[:,i].max() + 1
        bins[i] = binning
        
    return data_subset, bins


def learn_net_discretize(data, vars_to_discretize, n_bins):
    '''learn Bayes net after selected variables have been discretized'''
    data_subset, bins = discretize(data, vars_to_discretize, n_bins=n_bins)
    data_dict = data_subset.to_dict('records')
    learner = PGMLearner()
    skel = learner.discrete_constraint_estimatestruct(data=data_dict,indegree=1)
    skel.toporder()
    disc_bayes_net = learner.discrete_mle_estimateparams(graphskeleton=skel,data=data_dict)
    return disc_bayes_net, bins

def learn_net(data):
    '''learns Bayes net on raw data'''
    data_dict = data.to_dict('records')
    learner = PGMLearner()
    skel = learner.discrete_constraint_estimatestruct(data=data_dict,indegree=1)
    skel.toporder()
    disc_bayes_net = learner.discrete_mle_estimateparams(graphskeleton=skel,data=data_dict)
    return disc_bayes_net

def sample(net, bins, n_samples):
    '''
    sample initial snapshot of customers using the provided Bayes net and 
    information about the number of bins for each variable
    '''
    sampled_data = net.randomsample(n=n_samples)
    for j in bins:
        na_level = len(bins[j])
        for i in xrange(len(sampled_data)):
            ind = sampled_data[i][j]
            if ind == na_level - 1:
                sampled_data[i][j] = None
    return ps.DataFrame(sampled_data)

def granularize(sampled_data,bins):
    '''
    for the sampled data, label None variables and draw a continuous var. 
    from the range of each bin to make discrete variables continuous
    '''
    for j in bins:
        na_level = len(bins[j])

        for i in xrange(len(sampled_data)):
            ind = sampled_data[i][j]
            if ind == na_level - 1: # if the sampled value corresponds to None variable
                sampled_data[i][j] = None
            else:
                sampled_data[i][j] = round((np.random.uniform() * (bins[j][ind+1] - bins[j][ind]) + 
                                  bins[j][ind]))

def sample_granular(net, bins, n_samples):
    '''
    sampled discrete random variables from Bayes net and then make them continuous
    '''
    sampled_data = net.randomsample(n=n_samples)
    granularize(sampled_data,bins)
    return ps.DataFrame(sampled_data)

def graph_to_pdf(nodes, edges, name):
    '''
    save a plot of the Bayes net graph in pdf
    '''
    G=nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.drawing.nx_pydot.write_dot(G,name + ".dot")
    os.system("dot -Tpdf %s > %s" % (name+'.dot', name+'.pdf'))

# CREATE DICTIONARY TO HOLD RESULTS
record = dict()

# LOAD DATA
print('Loading data')
data = ps.read_csv("../kdd98_data/kdd1998tuples.csv", header=None)
data.columns = ['customer','period','r0','f0','m0','ir0','if0','gender','age','income',
                'zip_region','zip_la','zip_lo','a','rew','r1','f1','m1','ir1','if1',
                'gender1','age1','income1','zip_region1','zip_la1','zip_lo1']

cols = ['r0','f0','m0','ir0','if0','gender','age','income','zip_region']
data['age'][data['age'] == 0] = None

# PREPARE DATA
print('Preprocessing data')
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

train_samples = 50000
customers = list(set(data['customer']))
np.random.shuffle(customers)
selected_customers = np.array(customers[0:train_samples])

vars_to_discretize = {'m0':'Bins','age':'Bins'} # 'r0':'Bins','f0':'Bins','ir0':'Bins','if0':'Bins',
n_bins = {'m0':[-0.01, 0, 5, 10, 15, 20, 25, 50, 4000],
          'age':[17, 25, 40, 50, 60, 80, 100]}

# LEARN BAYES NET
print('Learning Bayes net')
data_subset = data[data['period'].isin([0,1,2,3]) & data['customer'].isin(selected_customers)][cols]
net_start_life, start_bins = learn_net_discretize(data_subset, vars_to_discretize, n_bins)
save(net_start_life,"../results/kdd98_init_snapshot_start.p")
save(start_bins,"../results/kdd98_init_snapshot_start_bins.p")
graph_to_pdf(net_start_life.V,net_start_life.E,"../results/kdd98_init_snapshot_start")

# Sample and validate initial snapshot of customers
print('Validate Bayes net')

data_subset = data[data['period'].isin([0,1,2,3]) & ~data['customer'].isin(selected_customers)][cols].sample(1000, random_state=RANDOM_SEED)
sampled_data = sample_granular(net=net_start_life, bins=start_bins, n_samples=1000)


plot_validate(data_subset["r0"], sampled_data['r0'], xlab="Recency", ylab=False, name="../results/kdd98_r0.pdf", 
              n_bins=4, x_range=(0,4), y_range=(0,1), font = 28)
plot_validate(data_subset["f0"], sampled_data['f0'], xlab="Frequency", ylab=False, name="../results/kdd98_f0.pdf", 
              n_bins=4, x_range=(0,4), y_range=(0,1), font = 28)
plot_validate(data_subset["m0"], sampled_data['m0'], xlab="Avg. Past Donation", ylab=False, name="../results/kdd98_m0.pdf", 
              n_bins=6, x_range=(0,30), y_range=(0,1), bar_width = 1.5, font = 28, legend=True)
plot_validate(data_subset["ir0"], sampled_data['ir0'], xlab="Interaction Recency", ylab=False, name="../results/kdd98_ir0.pdf", 
              n_bins=4, x_range=(0,4), y_range=(0,0.8), font = 28)
plot_validate(data_subset["if0"], sampled_data['if0'], xlab="Interaction Frequency", ylab=False, name="../results/kdd98_if0.pdf", 
              n_bins=4, x_range=(0,4), y_range=(0,0.8), font = 28)
plot_validate(data_subset["gender"], sampled_data['gender'], xlab="Gender", ylab=False, name="../results/kdd98_gender.pdf", 
              n_bins=4, x_range=(0,4), y_range=(0,0.8), font = 28)
plot_validate(data_subset["zip_region"], sampled_data['zip_region'], xlab="Zip Region", ylab=False, name="../results/kdd98_zip_region.pdf", 
              n_bins=10, x_range=(0,10), y_range=(0,0.5), font = 28)
plot_validate(data_subset["age"], sampled_data['age'], xlab="Age", ylab=False, name="../results/kdd98_age.pdf", 
              n_bins=10, x_range=(0,90), y_range=(0,0.5), bar_width = 3, font = 28)
plot_validate(data_subset["income"], sampled_data['income'], xlab="Income", ylab=False, name="../results/kdd98_income.pdf", 
              n_bins=10, x_range=(0,10), y_range=(0,0.5), font = 28)

record['KL_divergence_r'] = str(KL_validate(data_subset["r0"], sampled_data['r0'], n_bins=4, x_range=(0,4)))
record['KL_divergence_f'] = str(KL_validate(data_subset["f0"], sampled_data['f0'], n_bins=4, x_range=(0,4)))
record['KL_divergence_m'] = str(KL_validate(data_subset["m0"], sampled_data['m0'], n_bins=6, x_range=(0,30)))
record['KL_divergence_ir'] = str(KL_validate(data_subset["ir0"], sampled_data['ir0'], n_bins=4, x_range=(0,4)))
record['KL_divergence_if'] = str(KL_validate(data_subset["if0"], sampled_data['if0'], n_bins=4, x_range=(0,4)))
record['KL_divergence_age'] = str(KL_validate(data_subset["age"], sampled_data['age'], n_bins=10, x_range=(0,100)))
record['KL_divergence_zip'] = str(KL_validate(data_subset["zip_region"], sampled_data['zip_region'], n_bins=10, x_range=(0,10)))
record['KL_divergence_gender'] = str(KL_validate(data_subset["gender"], sampled_data['gender'], n_bins=3, x_range=(0,3)))
record['KL_divergence_income'] = str(KL_validate(data_subset["income"], sampled_data['income'], n_bins=8, x_range=(0,8)))

record['orig_means'] = str(data_subset.mean())
record['sim_means'] = str(sampled_data.mean())
record['orig_stds'] = str(data_subset.std())
record['sim_stds'] = str(sampled_data.std())

plot_validate_bivariate(data_subset["ir0"], data_subset["if0"], sampled_data['ir0'], sampled_data['if0'], n_bins=(10,10),
                        xlab="Interaction Recency", ylab="Interaction Frequency", 
                        name="../results/kdd98_bi_ir0_if0.pdf", 
                        x_range=(-2,5), y_range=(-2,5))
plot_validate_bivariate(data_subset["r0"], data_subset["f0"], sampled_data['r0'], sampled_data['f0'], n_bins=(10,10),
                        xlab="Donation Recency", ylab="Donation Frequency", 
                        name="../results/kdd98_bi_r0_f0.pdf", 
                        x_range=(-2,5), y_range=(-2,5), legend=True)
plot_validate_bivariate(data_subset["gender"], data_subset["income"], 
                        sampled_data['gender'], sampled_data['income'], n_bins=(10,10),
                        xlab="Gender", ylab="Income", 
                        name="../results/kdd98_bi_gender_income.pdf", 
                        x_range=(-2,5), y_range=(-2,10))
plot_validate_bivariate(data_subset["m0"], data_subset["age"], 
                        sampled_data['m0'], sampled_data['age'], n_bins=(10,10),
                        xlab="Avg. Past Donation", ylab="Age", 
                        name="../results/kdd98_bi_m0_age.pdf", 
                        x_range=(-5,30), y_range=(-2,90))

# SAVE RECORD
save_json(record,'../results/kdd98_record_initial_snapshot.json')
print(record)

