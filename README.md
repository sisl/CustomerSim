# CustomerSim

This repository contains code for `Customer simulation for direct marketing experiments` paper, [IEEE DSAA 2016](http://ieeexplore.ieee.org/document/7796934/). Authors: Yegor Tkachenko, Mykel J. Kochenderfer, Krzysztof Kluza.

# Software

Python is our main language (with some preprocessing done in R).

You will need the following Python packages

+ libpgm - for learning Bayesian networks (install from `https://github.com/CyberPoint/libpgm`)
+ Keras and Theano - for deep learning (make sure you have the cutting edge theano via `pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git`)
+ scikit-learn - for random forests

+ NumPy
+ SciPy
+ Matplotlib
+ pickle
+ networkx
+ json
+ pandas
+ Cython
+ H5PY

R prerequisites include

+ plyr
+ zipcode

# Data

+ KDD1998: `https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html` (accessed on 2015-07-04)
+ Valued Shoppers: `https://www.kaggle.com/c/acquire-valued-shoppers-challenge` (accessed on 2015-10-07)

# Getting started

+ `src` folder contains the code to replicate results of the paper
+ the folder should be placed within a parent folder that contains unzipped data, downloaded from the links above
+ then run `bash train_and_validate_all.sh` - it will take a couple of hours to finish (to preprocess the data, to train and validate the models)
+ all results will be stored in the `results` folder

Once the code has run, follow code in `simulate_kdd98.py` and `simulate_vs.py` to perform simulation based on KDD1998 and Kaggle Valued Shoppers data sets (respectively). You can also check out files in `src` folder for details on validation procedure.

Note, this code contains a correction and uses &tau;=0.275 in KDD98 simulation, instead of the &tau;=0.25 in the paper, leading to improved results in `KDD1998 simulation, 18 times step` section of Table II of the paper.

# Required directory structure

+ customersim
	+ train_and_validate_all.sh
	+ simulate_kdd98.py
	+ simulate_vs.py
	+ kdd98_data
		+ cup98LRN.txt
		+ cup98VAL.txt
	+ kaggle_valued_shoppers
		+ offers
		+ testHistory
		+ trainHistory
		+ transactions
	+ src
		+ net_designs.py
		+ shared_functions.py
		+ kdd98_preprocess.R
		+ kdd98_initial_snapshot.py
		+ kdd98_propagate_classifier.py
		+ kdd98_propagate_regressor.py
		+ kdd98_simulate.py
		+ vs_preprocess_category.py
		+ vs_propagate_regressor_category.py
		+ vs_simulate.py

# Technical note - recency metric

We take use of "transaction recency" and "interaction recency" metrics in both simulators, where by recency we mean the "number of periods elapsed since the last transaction" (or since the last interaction). This definition runs into a problem when we ask what should the value of these metrics be before the customer has transacted or has been interacted with. For this reason, it may be desirable to focus on analysis of "repeat" transactions only (effectively discarding the information about the preceding period), and so transaction recency has a clear interpretation. This approach, however, also runs into some problems. 

For example, if we are tracking, in parallel, customer transactions and marketing interactions with the customer, we may be in a situation where the customer has transacted several times, but we have not interacted with him once. At that point in time transaction recency metric would have a clear interpretation, but it would still be not clear what value to set the interaction recency metric to. And focusing only on repeat interactions, as often done in the literature, would lead us to potentially discard a lot of otherwise useful data.

An alternative approach, which we take in this work, is to redefine "transaction recency" not only as "time elapsed since the transaction", but also, if no transaction has yet occurred, as "time elapsed since the beginning of observation of the group of customers". The advantage of this approach is that it removes the need to recluse oneself to analysis of repeat events only and data discarding, alleviates the need to resolve conflicts between different metrics tracked in parallel, and also significantly simplifies the computation. 

The disadvantage is that transaction recency metric acquires different meanings for customers with frequency of transaction >0 and for customers with frequency of transaction = 0, which may lead to noise if we try to use the simulator to learn customer lifetime value in these parts of state space. However, while the learning may be slightly trickier, given that both frequency and recency of transactions are included in the customer state description, a good function approximator can learn  variation in customer value between those different states. (The above discussion applies to interaction recency too).


# License

The software is distributed under the Apache License 2.0
