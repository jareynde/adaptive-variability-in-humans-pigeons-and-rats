#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for pigeons
Pigeon data in seperate files per subject and condition

In this copy, we fit these models:
*3.2: eps-lr-gamnr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model
#model 3.2
def negll_RW_eps_lr_gamnr(params, k, r, buffer):

    eps, lr, gamnr = params

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


#    model 3.2  eps, lr, gamnr = params
results32 = {}
data_eps32 = np.zeros(15, dtype=float)
data_alpha32 = np.zeros(15, dtype=float)
data_gamnonr32 = np.zeros(15, dtype=float)
data_BIC32 = np.zeros(15, dtype=float)
data_negLL32 = np.zeros(15, dtype=float)
data_trials32 = np.zeros(15, dtype=float)



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


    #fit model 3.2
    #eps, lr, gamnr = params
    result32 = differential_evolution(negll_RW_eps_lr_gamnr, bounds=bounds3, args=(k,r,buffer), strategy=strategy, constraints=constraint3)
    negLL32 = result32.fun
    param_fits32= result32.x

    data_eps32[sub] = param_fits32[0]
    data_alpha32[sub] = param_fits32[1]
    data_gamnonr32[sub] = param_fits32[2]
    data_negLL32[sub] = negLL32
    data_trials32[sub] = trials

    BIC32 = len(bounds3) * np.log(trials) + 2*negLL32
    data_BIC32[sub] = BIC32
    print('model 3.2 done')


#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params


#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params







results32['eps'] = data_eps32
results32[f'alpha'] = data_alpha32
results32[f'gamnonr'] = data_gamnonr32
results32[f'BIC'] = data_BIC32
results32[f'negLL'] = data_negLL32
results32[f'trials'] = data_trials32
results32['Choices'] = choices
    
results32['sub'] = data_sub


for s in range(len(files)):

    line_to_write = {"sub": data_sub[s], "eps": data_eps32[s], "alpha": data_alpha32[s], "gamnr": data_gamnonr32[s], "BIC": data_BIC32[s], "negLL": data_negLL32[s], "trials": data_trials32[s], "choices": choices[s]}
    with open("Pigeon_M3_2.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "gamnr", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()