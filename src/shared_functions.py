# HELPER FUNCTIONS

import pandas as ps
import numpy as np
import pickle
import json
import math
import random

from matplotlib import rcParams
rcParams['font.family'] = 'Euclid'
import matplotlib.pyplot as plt

import os
from copy import deepcopy
from scipy import stats as sc

from sklearn.metrics import roc_curve, auc

if not os.path.isdir('../results'):
    os.mkdir('../results')



# INPUT-OUTPUT FUNCTIONS
def save(obj,name):
    pickle.dump(obj, open(name, "wb"))

def load(name):
    return pickle.load(open(name, "rb"))

def save_json(obj,name):
    with open(name, 'w') as outfile:
        json.dump(obj, outfile)

# PROBABILITY MASS FUNCTION
def plot_validate(data_true, data_predicted, xlab, ylab, name, n_bins, x_range, y_range, font = 15, legend = False, bar_width = 0.4):
    
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

# PERCENTILE KL DIVERGENCE BOOTSTRAP TEST
def KL_validate(data_true, data_predicted, n_bins, x_range, n_samples=10000):
    '''"Pr(KL(simulated data||original) > KL(bootstrap original||bootstrap original))'''

    n = data_true.shape[0]

    hist_true, _ = np.histogram(data_true, bins=n_bins, range=x_range)
    hist_predicted, bin_edges = np.histogram(data_predicted, bins=n_bins, range=x_range)
    
    simulated_KL = sc.entropy(hist_true+1,hist_predicted+1)
    subsampled_KL = []

    for i in xrange(n_samples):
        index1 = np.random.choice(n, n, replace=True)
        index2 = np.random.choice(n, n, replace=True)
        sample1 = data_true[index1]
        sample2 = data_true[index2]
        hist_sample1, _ = np.histogram(sample1, bins=n_bins, range=x_range)
        hist_sample2, _ = np.histogram(sample2, bins=n_bins, range=x_range)
        subsampled_KL.append(sc.entropy(hist_sample2+1,hist_sample1+1))
    
    subsampled_KL = sorted(subsampled_KL)
    pval = sum( simulated_KL < i for i in subsampled_KL) / float(n_samples)
    conf_interval = (0,subsampled_KL[int(math.ceil(n_samples*0.95))-1])
    return simulated_KL,conf_interval,pval,n

# CONTOUR PLOTS
def plot_validate_bivariate(data_true_a, data_true_b, data_predicted_a, data_predicted_b, 
                            n_bins, xlab, ylab, name, x_range, y_range, legend=False):
    
    H_true, xedges_true, yedges_true = np.histogram2d(data_true_a + np.random.normal(0,0.4,data_true_a.shape), 
                                       data_true_b + np.random.normal(0,0.4,data_true_b.shape), 
                                       range=[x_range, y_range], bins=n_bins)
    
    H_true = H_true/H_true.sum()
    
    H_predicted, xedges_predicted, yedges_predicted = np.histogram2d(data_predicted_a + np.random.normal(0,0.4,data_predicted_a.shape), 
                                       data_predicted_b + np.random.normal(0,0.4,data_predicted_b.shape), 
                                       range=[x_range, y_range], bins=n_bins)
    
    H_predicted = H_predicted/H_predicted.sum()
    
    extent_true = [yedges_true[0], yedges_true[-1], xedges_true[0], xedges_true[-1]]
    extent_predicted = [yedges_predicted[0], yedges_predicted[-1], xedges_predicted[0], xedges_predicted[-1]]
    
    plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
    
    levels = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02)
    plt.contour(H_true, levels, origin='lower', colors=['green'], 
                        linewidths=(1, 1, 1, 1, 1, 1, 1),linestyles=['solid'],extent=extent_true,label="Actual Data")
    plt.contour(H_predicted, levels, origin='lower', colors=['brown'],linestyles=['dashed'],
                    linewidths=(1, 1, 1, 1, 1, 1, 1),
                    extent=extent_predicted,label = "Simulated Data")
    
    leg1, = plt.plot([],[], label="Actual Data", color="green")
    leg2, = plt.plot([],[], label="Simulated Data", color="brown", linestyle='dashed')
    if legend:
        plt.legend(handles=[leg1,leg2], fontsize=28)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    
    plt.xlabel(ylab,fontsize=28, labelpad=15)
    plt.ylabel(xlab,fontsize=28, labelpad=15)
    plt.xlim(y_range[0], y_range[1])
    plt.ylim(x_range[0], x_range[1])

    plt.savefig(name, bbox_inches='tight')
    plt.close()


# Compute ROC curve and ROC area for each class
def roc(y_label,y_score,name):

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    ind_max = np.argmax(1 - fpr[1] + tpr[1])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='w')
    plt.plot(fpr[1], tpr[1], label='ROC Curve (Area = %0.2f)' % roc_auc[1],color="g")

    plt.plot([fpr[1][ind_max], fpr[1][ind_max]], [fpr[1][ind_max], tpr[1][ind_max]], 'k:')
    plt.annotate(r'$\bf J$', xy=(fpr[1][ind_max]-0.04, (fpr[1][ind_max] + tpr[1][ind_max])/2), color='black', 
             fontsize=20)

    plt.plot(fpr[1][ind_max], tpr[1][ind_max], marker ='v', markersize=10, linestyle='None', color='brown', 
         label="Decision Threshold (DT),\nMax. Youden's J Statistic")
    plt.annotate('DT: %0.2f\nTPR: %0.2f\nFPR: %0.2f' % (thresholds[1][ind_max], tpr[1][ind_max], fpr[1][ind_max]), 
             xy=(fpr[1][ind_max]+0.015, tpr[1][ind_max]-0.175), color='black', fontsize=20)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.xlabel('False Positive Rate (1 - Specificity)',fontsize=20, labelpad=15)
    plt.ylabel('True Positive Rate (Sensitivity)',fontsize=20, labelpad=15)
    plt.legend(loc="lower right",fontsize=20,numpoints=1)
    plt.savefig(name, bbox_inches='tight')
    plt.close()
