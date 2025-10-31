#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for pigeons
Pigeon data in seperate files per subject and condition

In this copy, we fit these models:
*5: eps-lr-uc-ucs

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model
#model 5
def negll_RW_eps_lr_uc_ucs(params, k, r):

    eps, lr, uc, ucs, w = params
    Q_int = 1
    K = np.max(k)+1
    K_seq = K*K
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    S_k = np.ones(K_seq)*Q_int
    S_k_stored = np.zeros((T,K_seq), dtype=float)

    seq_options = np.array([[a, b] for a in range(K) for b in range(K)])

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        Q_k_stored[t,:] = Q_k
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)

        combined_QandS_info = Q_k + w*next_seq_options
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_C = np.argmax(combined_QandS_info)  # Find the option with the maximum Combined value
        p[max_C] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)
        
        if t != 0:
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
        

            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index] -= ucs 

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += lr * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


constraint3 = LinearConstraint([1,0,1], 0, 1)
constraint4 = LinearConstraint([1,0,0,1], 0, 1)
constraint7 = LinearConstraint([1, 0, 0, 0, 0, 1], 0, 1)


Q_int = 1


strategy = 'randtobest1bin'

rewards = {}
mean_rewards = np.zeros(15)
standarderror_rewards = np.zeros(15)

data_sub = np.zeros(15,dtype=float)


#    model 5    eps, lr, uc, ucs, w = params

results5 = {}
data_eps5 = np.zeros(15, dtype=float)
data_alpha5 = np.zeros(15, dtype=float)
data_unchosen5 = np.zeros(15, dtype=float)
data_unchosenseq5 = np.zeros(15,dtype=float)
data_weight5 = np.zeros(15, dtype=float)
data_BIC5 = np.zeros(15, dtype=float)
data_negLL5 = np.zeros(15, dtype=float)
data_trials5 = np.zeros(15, dtype=float)


choices = np.zeros(15,dtype=float)

#Read Neuringer's data
files = ['sub1con2.csv', 'sub2con2.csv', 'sub3con2.csv', 'sub4con2.csv', 'sub5con2.csv', 'sub1con4.csv', 'sub2con4.csv', 'sub3con4.csv', 'sub4con4.csv', 'sub5con4.csv', 'sub1con8.csv', 'sub2con8.csv', 'sub3con8.csv', 'sub4con8.csv', 'sub5con8.csv']

for sub, file in enumerate(files):
    df = pd.read_csv(f"AnimalModelFit/ALL_DATA_CSV/Pigeon/{file}")
    sub_data = df.to_numpy()

    trials = np.shape(sub_data)[0]
    k= sub_data[:,5]
    k= np.add(k,-1)
    k= k.astype(int)
            
    r= sub_data[:,6]
    r= r.astype(int)
    
    K = np.max(k)+1

    
    bounds1=[(0,1), (0,1)]
    bounds2=[(0,1), (0,1),(-1/K,1/K)]
    bounds3=[(0,1), (0,1), (0,1)]
    bounds4=[(0,1), (0,1), (-1/K,1/K), (0,1)]
    bounds5=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,10)]
    bounds7=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,1), (0,10)]

    buffer = K/2
    buffer = buffer.astype(int)

    

    choices[sub] = sub_data[10,2]
    seedings = int(np.sum(k[5:10])*trials)

    #random.seed(seedings)
    #np.random.seed(seedings)


#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params

#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params


    data_sub[sub] = sub

    #fit model 5
    #eps, lr, uc, ucs, w = params
    result5 = differential_evolution(negll_RW_eps_lr_uc_ucs, bounds=bounds5, args=(k,r), strategy=strategy)
    negLL5 = result5.fun
    param_fits5 = result5.x

    data_alpha5[sub] = param_fits5[0]
    data_eps5[sub] = param_fits5[1]
    data_unchosen5[sub] = param_fits5[2]
    data_unchosenseq5[sub] = param_fits5[3]
    data_weight5[sub] = param_fits5[4]
    data_negLL5[sub] = negLL5
    data_trials5[sub] = trials

    BIC5 = len(bounds5) * np.log(trials) + 2*negLL5
    data_BIC5[sub] = BIC5
    print('model 5 done')


   

    

#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params


#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params




results5['eps'] = data_eps5
results5[f'alpha'] = data_alpha5
results5[f'unchosen'] = data_unchosen5
results5[f'unchosenseq'] = data_unchosenseq5
results5[f'weight'] = data_weight5
results5[f'BIC'] = data_BIC5
results5[f'negLL'] = data_negLL5
results5[f'trials'] = data_trials5
results5['Choices'] = choices
    

results5['sub'] = data_sub


for s in range(len(files)):

    line_to_write = {"sub": data_sub[s], "eps": data_eps5[s], "alpha": data_alpha5[s], "lambda": data_unchosen5[s], "lambda_s": data_unchosenseq5[s], "w": data_weight5[s], "BIC": data_BIC5[s], "negLL": data_negLL5[s], "trials": data_trials5[s], "choices": choices[s]}
    with open("Pigeon_M5.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "lambda", "lambda_s", "w", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()