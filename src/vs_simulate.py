
# VS SIMULATE

# from shared_functions import *
# from net_designs import *

import pandas as ps
import json
import numpy as np
import pickle
from joblib import dump, load
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as sct
import math
from scipy import stats as sc
import random
from sklearn.ensemble import RandomForestRegressor
import h5py
import os
from time import time
from copy import deepcopy

from scipy import stats as sc
from tqdm import tqdm
RANDOM_SEED = 999
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class Simulator(object):
    def __init__(self, model_path: str):
        self.record = dict()
        self.load_model(model_path)
        
    def random_policy(self, data):
        # The company selects:
        #  the number of offers in each category
        #  the average min. number of goods required to activate the offer
        #  and the average offer value.
        shape = (data.shape[0], data.shape[1],1)
        offer_prob = np.random.uniform(0, 1, shape)
        has_offer = 1*(offer_prob<0.05)
        vector = (has_offer, has_offer, 5*has_offer)
        return np.concatenate(vector, axis=2)

    def init_variables(self, periods, num_actions, data):
        shape = (periods, data.shape[0], data.shape[1])
        actions = np.zeros(tuple( list(shape) + [num_actions] ),
                           dtype = np.float32)
        quantity = np.zeros(shape, dtype = np.float32)
        amount = np.zeros(shape, dtype = np.float32)
        return actions, quantity, amount

    def init_customers(self, periods, data):
        shape = (periods+1, data.shape[0], data.shape[1], data.shape[2])
        customers = np.zeros(shape, dtype = np.float32)
        customers[0] = data
        return customers
    
    def init_arrays(self, periods, num_actions, data):
        customers = self.init_customers(periods, data)
        actions, quantity, amount = self.init_variables(periods, num_actions, data)
        return customers, actions, quantity, amount
    
    def select_action(self, orig_actions, policy, customer):
        # SELECTING ACTIONS - IF A TENSOR OF ORIGINAL ACTIONS IS PROVIDED, IGNORE POLICY
        if isinstance(orig_actions, (np.ndarray)):
            action = orig_actions
        # OTHERWISE, USE POLICY
        else:
            action = policy(customer)
        return action
    
    def concat_reshape(self, customer, action):
        inp = np.append(customer, action, axis = 2).astype(np.float32)
        inp = inp.reshape(inp.shape[0],-1)
        return inp

    def predict_lambda(self, inp, regressor):
        quantity_pred = regressor.predict(inp)
        quantity_pred = np.rint(quantity_pred).astype(np.float32)
        quantity_pred[quantity_pred < 0] = 0
        return quantity_pred
    
    def sim_quantity(self, quantity_pred):
        return np.random.poisson(quantity_pred,quantity_pred.shape)

    def predict_quantity(self, inp, regressor):
        quantity_pred = self.predict_lambda(inp, regressor)
        quantity_sim = self.sim_quantity(quantity_pred)
        return quantity_sim

    def get_amount(self, quantity, prices):
        return quantity * np.tile(prices, (quantity.shape[0],1))

    def update_states(self, customers, quantity, amount, actions, t):
        # Recency
        customers[t+1,:,:,0] = (customers[t,:,:,0] + 1)*(quantity[t] == 0)
        # Frequency
        customers[t+1,:,:,1] = customers[t,:,:,1] + quantity[t]
        # Avg. Past Value
        customers[t+1,:,:,2] = (customers[t,:,:,2] * customers[t,:,:,1] + amount[t]) / (customers[t+1,:,:,1] + 1*(customers[t+1,:,:,1]==0))
        # Offer Recency
        customers[t+1,:,:,3] = (customers[t,:,:,3] + 1)*(actions[t,:,:,0] == 0) # Null action 1
        # Offer Frequency
        customers[t+1,:,:,4] = customers[t,:,:,4] + actions[t,:,:,0]
        return customers

    def propagate(self, data, regressor, policy, prices, periods=12, num_actions=3, orig_actions=None):
        # Initializing arrays to hold output
        customers, actions, quantity, amount = self.init_arrays(periods, num_actions, data)
        for t in tqdm(range(periods), desc="Propagating"):
            if orig_actions:
                actions[t] = self.select_action(orig_actions[t], policy, customers[t])
            else:
                actions[t] = self.select_action(None, policy, customers[t])
            # Build a vector of shape [N_customers, (categories * (S+A))]
            inp = self.concat_reshape(customers[t], actions[t])
            # PROPAGATING CUSTOMERS
            quantity[t] = self.predict_quantity(inp, regressor)
            amount[t] = self.get_amount(quantity[t], prices)
            # UPDATING CUSTOMER STATE
            customers = self.update_states(customers, quantity, amount, actions, t)
        return customers, actions, quantity, amount

    def load_model(self, path: str):
        print('Loading')
        self.regressor = load(path)

    def load_data(self, data_path: str, prices_path: str):
        h5f = h5py.File(data_path,'r')
        data = h5f['temp_data'][:]
        h5f.close()
        self.columns = ['transaction_recency','transaction_frequency','avg_past_transaction_value',
                        'offer_recency','offer_frequency','number_of_offers','offer_goods_quantity_per_offer','offer_value_per_offer',
                        'purchased_goods_quantity','purchased_item_price']
        self.cols_X = [0,1,2,3,4,5,6,7]
        self.cols_S = [0,1,2,3,4]
        self.cols_A = [5,6,7]
        self.cols_Y = 8
        self.cols_Z = 9
        # LOAD AVG PRICES PER CATEGORY
        prices = load(prices_path)
        return data, prices

    def split_data(self, data):
        rand_ind = np.random.permutation(data.shape[0])
        test_customers = rand_ind[216000:217000] # 1000 customers from test set
        test_data = data[test_customers].T
        # EXTRACT ARRAYS WITH ORIGINAL DATA
        orig_S = test_data[self.cols_S].T          # Recency, Frequency, Avg past value, Offer recency, offer frequency
        orig_A = test_data[self.cols_A].T          # Number of offers, Offer goods quantity, Offer value
        orig_Q = test_data[self.cols_Y].T          # Purchased quantity
        orig_P = test_data[self.cols_Z].T * orig_Q # Purchased value
        orig_S = np.transpose(orig_S, (1, 0, 2, 3))
        orig_A = np.transpose(orig_A, (1, 0, 2, 3))
        orig_Q = np.transpose(orig_Q, (1, 0, 2))
        orig_P = np.transpose(orig_P, (1, 0, 2))
        return orig_S, orig_A, orig_Q, orig_P

    def simulate(self, data_path: str = 'data/compressed/temp_data_cat.h5', prices_path: str = 'data/compressed/vs_cat_avg_prices.p', action_type: str = 'original'):
        # SIMULATE DATA
        # INPUT IS THE STARTING STATE ARRAY, MODEL, TESTED POLICY [IGNORED BECUASE WE PROVIDE ACTIONS FOR EVERY TIME PERIOD AS ORIG ACTIONS]
        # AVG. PRICES FOR EACH CATEGORY, PERIODS FOR SMULATION, NUMBER OF VARIABLES DESCRIBING THE ACTION, [OPTIONAL] TENSOR OF ORIGINAL ACTIONS 
        data, prices = self.load_data(data_path, prices_path)
        orig_S, orig_A, orig_Q, orig_P = self.split_data(data)
        if action_type == 'original':
            S, A, Q, P = self.propagate(orig_S[0], self.regressor, self.random_policy, prices, periods=16, num_actions=3, orig_actions=orig_A)
        elif action_type == 'random':
            S, A, Q, P = self.propagate(orig_S[0], self.regressor, self.random_policy, prices, periods=16, num_actions=3)
        else:
            raise Exception(f"Invalid action_type: {action_type}")
        self.record_key_metrics(P, orig_P, S, orig_S)
        self.save_record(f'data/simulation_{time()}_records.json')

        simulation = {'S': S, 'A': A, 'Q': Q, 'P': P}
        original = {'orig_S': orig_S, 'orig_A': orig_A, 'orig_Q': orig_Q, 'orig_P': orig_P}
        return simulation, original
    
    def get_simulated_kl(self, data_true, data_predicted, n_bins, x_range):
        n = data_true.shape[0]
        hist_true, _ = np.histogram(data_true, bins=n_bins, range=x_range)
        hist_predicted, _ = np.histogram(data_predicted, bins=n_bins, range=x_range)
        simulated_KL = sc.entropy(hist_true+1,hist_predicted+1)
        return simulated_KL, n

    def get_subsampled_kl(self, data_true, n, n_bins, x_range):
        index1 = np.random.choice(n, n, replace=True)
        index2 = np.random.choice(n, n, replace=True)
        sample1 = data_true[index1]
        sample2 = data_true[index2]
        hist_sample1, _ = np.histogram(sample1, bins=n_bins, range=x_range)
        hist_sample2, _ = np.histogram(sample2, bins=n_bins, range=x_range)
        return sc.entropy(hist_sample2+1,hist_sample1+1)

    def pval(self, simulated_KL, subsampled_KL, n_samples):
        return sum( simulated_KL < i for i in subsampled_KL) / float(n_samples)
    
    def conf_interval(self,  subsampled_KL, n_samples):
        return (0,subsampled_KL[int(math.ceil(n_samples*0.95))-1])

    def get_pval_and_conf(self, subsampled_KL, simulated_KL, n_samples):
        subsampled_KL = sorted(subsampled_KL)
        pval = self.pval(simulated_KL, subsampled_KL, n_samples)
        conf_interval = self.conf_interval(subsampled_KL, n_samples)
        return pval, conf_interval

    # PERCENTILE KL DIVERGENCE BOOTSTRAP TEST
    def KL_validate(self, data_true, data_predicted, n_bins, x_range, n_samples=10000):
        '''"Pr(KL(simulated data||original) > KL(bootstrap original||bootstrap original))'''
        simulated_KL, n = self.get_simulated_kl(data_true, data_predicted, n_bins, x_range)
        subsampled_KL = []
        for _ in range(n_samples):
            kl = self.get_subsampled_kl(data_true, n, n_bins, x_range)
            subsampled_KL.append(kl)
        pval, conf_interval = self.get_pval_and_conf(subsampled_KL, simulated_KL, n_samples)
        return simulated_KL,conf_interval,pval,n

    def record_purchases_KL(self, P, orig_P):
        significant_diffs = 0
        for i in tqdm(range(20), desc='Bootstrapping KL'):
            simulated_KL,conf_interval,pval,n = self.KL_validate(orig_P.sum(0)[:,i].squeeze(),
                                                                 P.sum(0)[:,i].squeeze(),
                                                                 n_bins=7,
                                                                 x_range=(0,525))
            self.record['KL_purchases_'+str(i)] = str((simulated_KL,conf_interval,pval,n))
            significant_diffs+= 1*(pval<0.05)
        self.record['KL_purchases_purchases_significant'] = int(significant_diffs)

    def record_purchases(self, P, orig_P):
        orig_purchases = np.sum(orig_P.sum(0),1).squeeze()
        sim_purchases = np.sum(P.sum(0),1).squeeze()
        self.record['KL_divergence_deeplearning_purchases'] = str(self.KL_validate(orig_purchases, sim_purchases, n_bins=7, x_range=(0,525)))
        self.record['orig_mean_deeplearning_purchases'] = str(np.mean(orig_purchases))
        self.record['sim_mean_deeplearning_purchases'] = str(np.mean(sim_purchases))
        self.record['orig_std_deeplearning_purchases'] = str(np.std(orig_purchases))
        self.record['sim_std_deeplearning_purchases'] = str(np.std(sim_purchases))
        self.record_purchases_KL(P, orig_P)

    def record_recency(self, S, orig_S):
        # 16 == last period
        orig_recen = np.mean(orig_S[16],1)[:,0].squeeze()
        sim_recen = np.mean(S[16],1)[:,0].squeeze()
        self.record['KL_divergence_deeplearning_recen'] = str(self.KL_validate(orig_recen, sim_recen, n_bins=5, x_range=(0,20)))
        self.record['orig_mean_deeplearning_recen'] = str(np.mean(orig_recen))
        self.record['sim_mean_deeplearning_recen'] = str(np.mean(sim_recen))
        self.record['orig_std_deeplearning_recen'] = str(np.std(orig_recen))
        self.record['sim_std_deeplearning_recen'] = str(np.std(sim_recen))

    def record_key_metrics(self, P, orig_P, S, orig_S):
        # MAKE A RECORD OF KEY PURCHASE METRICS
        self.record_purchases(P, orig_P)
        # CALCULATE THE NUMBER OF SIGNIFICANTLY DIFFERENT SIMULATED TOTAL PURCHASE HISOTGRAMS - BY CATEGORY
        self.record_recency(S, orig_S)

    def save_json(self, obj,name):
        with open(name, 'w') as outfile:
            json.dump(obj, outfile)

    def save_record(self, path):
        # SAVE RECORD
        self.save_json(self.record, path)
        print(self.record)


class Plots(object):
    def __init__(self, simuation: dict, original: dict):
        self.S=simuation['S']
        self.A=simuation['A']
        self.Q=simuation['Q']
        self.P=simuation['P']
        self.orig_S=original['orig_S']
        self.orig_A=original['orig_A']
        self.orig_Q=original['orig_Q']
        self.orig_P=original['orig_P']

    def plot_cumulative_purchases_by_category_over_time(self):
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
        for i in range(self.Q.shape[2]):
            x = range(self.orig_Q.shape[0])
            y = np.cumsum(self.orig_P.mean(1),0)[:,i]
            plt.plot(x, y, linewidth=2, alpha=0.3, color="green")

            x = range(self.Q.shape[0])
            y = np.cumsum(self.P.mean(1),0)[:,i]
            plt.plot(x, y, linewidth=2, alpha=0.5, color="brown",linestyle='--')
        plt.xlim(0,15)
        plt.ylim(0,50)
        line_green, = plt.plot([],[], label='Actual Data', color="green")
        line_brown, = plt.plot([],[], label='Simulated Data', color="brown",linestyle='--')
        #plt.legend(handles=[line_green, line_brown], fontsize=20)
        plt.xlabel("Campaign Period", fontsize=20, labelpad=15)
        plt.ylabel("Mean Cumulative Purchases", fontsize=20, labelpad=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)
        plt.savefig(f"data/results/vs_cumulative_purchase_{time()}.pdf", bbox_inches='tight')
        plt.close()

    # PROBABILITY MASS FUNCTION
    def plot_validate(self, data_true, data_predicted, xlab, ylab, name, n_bins, x_range, y_range, font = 15, legend = False, bar_width = 0.4):
        
        hist_true, bin_edges = np.histogram(data_true, bins=n_bins, range=x_range)
        hist_predicted, bin_edges = np.histogram(data_predicted, bins=n_bins, range=x_range)
        hist_true = hist_true / float(sum(hist_true))
        hist_predicted = hist_predicted / float(sum(hist_predicted))
        
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
        plt.bar(bin_edges[:-1],hist_true, bar_width,color="#60BD68",label='Actual Data')
        plt.bar(bin_edges[:-1]+bar_width,hist_predicted,bar_width,color="#FAA43A",alpha=1,label='Simulated Data') 
        plt.xlabel(xlab, fontsize=font, labelpad=15)
        if ylab:
            plt.ylabel(ylab, fontsize=font, labelpad=15)
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        
        xt_val = list(set([int(e) for e in bin_edges[:-1]]))
        xt_pos = [float(e) + bar_width for e in xt_val]
        
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)
        
        plt.xticks(xt_pos, xt_val)
        if legend:
            plt.legend(fontsize=font)
        plt.savefig(name, bbox_inches='tight')
        plt.close()
        
    def plot_histogram_of_total_purchases(self):
        orig_purchases = np.sum(self.orig_P.sum(0),1).squeeze()
        sim_purchases = np.sum(self.P.sum(0),1).squeeze()
        self.plot_validate(orig_purchases, sim_purchases,
                           xlab="Total Purchase Amount", ylab="Probability Mass", name=f"data/results/vs_total_purchase_{time()}.pdf",
                           n_bins=7, x_range=(0,525), y_range=(0,0.5),
                           font = 20, legend=True, bar_width=15)

    def plot_mean_recency_over_time(self):
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
        for i in range(self.S.shape[2]):
            plt.plot(range(self.orig_S.shape[0]), self.orig_S.mean(1)[:,i,0], linewidth=2, alpha=0.3, color="green")
            plt.plot(range(self.S.shape[0]), self.S.mean(1)[:,i,0], linewidth=2, alpha=0.5, color="brown",linestyle='--')
        plt.xlim(0,15)
        plt.ylim(0,20)
        line_green, = plt.plot([],[], label='Actual Data', color="green")
        line_brown, = plt.plot([],[], label='Simulated Data', color="brown",linestyle='--')
        plt.legend(handles=[line_green, line_brown], fontsize=20)
        plt.xlabel("Campaign Period", fontsize=20, labelpad=15)
        plt.ylabel("Mean Transaction Recency", fontsize=20, labelpad=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)
        plt.savefig(f"data/results/vs_recency_{time()}.pdf", bbox_inches='tight')
        plt.close()

    def plot_histogram_of_endperiod_recency(self):
        orig_recen = np.mean(self.orig_S[16],1)[:,0].squeeze()
        sim_recen = np.mean(self.S[16],1)[:,0].squeeze()
        self.plot_validate(orig_recen,sim_recen,
                           xlab="End-Period Recency", ylab="Probability Mass", name=f"data/results/vs_endperiod_recency_{time()}.pdf", 
                           n_bins=5, x_range=(0,20), y_range=(0,0.5), font = 20, legend=True, bar_width=1)
if __name__=='__main__':
    simulator = Simulator('data/models/rf_1625768815.117125')
    simulation, original = simulator.simulate(action_type='random')
    plots = Plots(simulation, original)
    plots.plot_cumulative_purchases_by_category_over_time()
    plots.plot_histogram_of_endperiod_recency()
    plots.plot_histogram_of_total_purchases()
    plots.plot_mean_recency_over_time()