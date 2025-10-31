#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for pigeons
Pigeon data in seperate files per subject and condition

In this copy, we fit these models:
*4.2: eps-lr-uc-gamnr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model
#model 4.2
def negll_RW_eps_lr_uc_gamnr(params, k, r, buffer):

    eps, lr, uc, gamnr = params

    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        Q_k_stored[t,:] = Q_k

        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p = np.zeros(K, dtype=float)

        p[max_Q] += 1-eps-gamnr
        p += eps/K
        
        if t >= buffer:
            options = [x for x in range(K) if x not in k[t-buffer:t]]
            for i in range(len(options)):
                #take a recent option from buffer
                p[options[i]] += gamnr/len(options)

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)

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


#   model 4.2  eps, lr, uc, gamnr = params
results42 = {}
data_eps42 = np.zeros(15, dtype=float)
data_alpha42 = np.zeros(15, dtype=float)
data_unchosen42 = np.zeros(15, dtype=float)
data_gamnonr42 = np.zeros(15, dtype=float)
data_BIC42 = np.zeros(15, dtype=float)
data_negLL42 = np.zeros(15, dtype=float)
data_trials42 = np.zeros(15, dtype=float)



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


    #fit model 4.2
    #eps, lr, uc, gamnr = params
    result42 = differential_evolution(negll_RW_eps_lr_uc_gamnr, bounds=bounds4, args=(k,r,buffer), strategy=strategy, constraints=constraint4)
    negLL42 = result42.fun
    param_fits42 = result42.x

    data_eps42[sub] = param_fits42[0]
    data_alpha42[sub] = param_fits42[1]
    data_unchosen42[sub] = param_fits42[2]
    data_gamnonr42[sub] = param_fits42[3]
    data_negLL42[sub] = negLL42
    data_trials42[sub] = trials

    BIC42 = len(bounds4) * np.log(trials) + 2*negLL42
    data_BIC42[sub] = BIC42
    print('model 4.2 done')



#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params


#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params



results42['eps'] = data_eps42
results42[f'alpha'] = data_alpha42
results42[f'unchosen'] = data_unchosen42
results42[f'gamnonr'] = data_gamnonr42
results42[f'BIC'] = data_BIC42
results42[f'negLL'] = data_negLL42
results42[f'trials'] = data_trials42
results42['Choices'] = choices


results42['sub'] = data_sub


for s in range(len(files)):

    line_to_write = {"sub": data_sub[s], "eps": data_eps42[s], "alpha": data_alpha42[s], "lambda" : data_unchosen42[s], "gamnr": data_gamnonr42[s], "BIC": data_BIC42[s], "negLL": data_negLL42[s], "trials": data_trials42[s], "choices": choices[s]}
    with open("Pigeon_M4_2.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "lambda", "gamnr", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()