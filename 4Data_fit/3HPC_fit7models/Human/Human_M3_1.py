#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Human data in seperate files per subject and condition

In this copy, we fit these models:
*3.1: eps-lr-gamr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model

#model 3.1
def negll_RW_eps_lr_gamr(params, k, r, buffer):

    eps, lr, gamr = params

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

    
        p[max_Q] += 1-eps-gamr
        p += eps/K

        if t >= buffer:
            buffer_content = k[t-buffer:t]
            for i in range(buffer):
                #take a recent option from buffer
                p[buffer_content[i]] += gamr/buffer


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
mean_rewards = np.zeros(18)
standarderror_rewards = np.zeros(18)

data_sub = np.zeros(18,dtype=float)

#    model 3.1  eps, lr, gamr = params
results31 = {}
data_eps31 = np.zeros(18, dtype=float)
data_alpha31 = np.zeros(18, dtype=float)
data_gamr31 = np.zeros(18, dtype=float)
data_BIC31 = np.zeros(18, dtype=float)
data_negLL31 = np.zeros(18, dtype=float)
data_trials31 = np.zeros(18, dtype=float)


choices = np.zeros(18,dtype=float)

#Read Neuringer's data
files = ['sub1con2.csv', 'sub2con2.csv', 'sub3con2.csv', 'sub4con2.csv', 'sub5con2.csv','sub6con2.csv', 'sub1con4.csv', 'sub2con4.csv', 'sub3con4.csv', 'sub4con4.csv', 'sub5con4.csv', 'sub6con4.csv', 'sub1con8.csv', 'sub2con8.csv', 'sub3con8.csv', 'sub4con8.csv', 'sub5con8.csv', 'sub6con8.csv']

for sub, file in enumerate(files):

    #data_dir1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #data_dir2 = os.path.join(data_dir1, 'ALL_DATA_CSV')
    #data_dir = os.path.join(data_dir2, 'Human')
    #df = pd.read_csv(os.path.join(data_dir, file))

    df = pd.read_csv(f"AnimalModelFit/ALL_DATA_CSV/Human/{file}")
    sub_data = df.to_numpy()

    trials = np.shape(sub_data)[0]
    k= sub_data[:,4]
    k= np.add(k,-1)
    k= k.astype(int)
            
    r= sub_data[:,5]
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

    


    #fit model 3.1
    #eps, lr, gamr = params
    result31 = differential_evolution(negll_RW_eps_lr_gamr, bounds=bounds3, args=(k,r,buffer), strategy=strategy, constraints=constraint3)
    negLL31 = result31.fun
    param_fits31 = result31.x

    data_eps31[sub] = param_fits31[0]
    data_alpha31[sub] = param_fits31[1]
    data_gamr31[sub] = param_fits31[2]
    data_negLL31[sub] = negLL31
    data_trials31[sub] = trials

    BIC31 = len(bounds3) * np.log(trials) + 2*negLL31
    data_BIC31[sub] = BIC31
    print('model 3.1 done')


    


#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params


#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params




results31['eps'] = data_eps31
results31[f'alpha'] = data_alpha31
results31[f'gamr'] = data_gamr31
results31[f'BIC'] = data_BIC31
results31[f'negLL'] = data_negLL31
results31[f'trials'] = data_trials31
results31['Choices'] = choices

results31['sub'] = data_sub



for s in range(len(files)):

    line_to_write = {"sub": data_sub[s], "eps": data_eps31[s], "alpha": data_alpha31[s], "gamr": data_gamr31[s], "BIC": data_BIC31[s], "negLL": data_negLL31[s], "trials": data_trials31[s], "choices": choices[s]}
    with open("Human_M3_1.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "gamr", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()
        