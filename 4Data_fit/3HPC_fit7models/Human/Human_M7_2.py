#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Human data in seperate files per subject and condition

In this copy, we fit these models:
*7_2: eps-lr-uc-ucs-gamnr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model
#model 7.2
def negll_RW_eps_lr_uc_ucs_gamnr(params, k, r, buffer):

    eps, lr, uc, ucs, w, gamnr = params
    

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

        max_Q = np.argmax(combined_QandS_info)  # Find the option with the maximum Q value
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
mean_rewards = np.zeros(18)
standarderror_rewards = np.zeros(18)

data_sub = np.zeros(18,dtype=float)

#    model 7.2  eps, lr, uc, ucs, w, gamnr = params
results72= {}
data_eps72 = np.zeros(18, dtype=float)
data_alpha72 = np.zeros(18, dtype=float)
data_unchosen72 = np.zeros(18, dtype=float)
data_unchosenseq72 = np.zeros(18,dtype=float)
data_weight72 = np.zeros(18, dtype=float)
data_gamnonr72 = np.zeros(18, dtype=float)
data_BIC72 = np.zeros(18, dtype=float)
data_negLL72 = np.zeros(18, dtype=float)
data_trials72 = np.zeros(18, dtype=float)




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
    bounds7=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,10), (0,1)]

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

    #fit model 7.2
    #eps, lr, uc, ucs, w, gamnr = params
    result72 = differential_evolution(negll_RW_eps_lr_uc_ucs_gamnr, bounds=bounds7, args=(k,r,buffer), strategy=strategy, constraints=constraint7)
    negLL72 = result72.fun
    param_fits72 = result72.x

    data_eps72[sub] = param_fits72[0]
    data_alpha72[sub] = param_fits72[1]
    data_unchosen72[sub] = param_fits72[2]
    data_unchosenseq72[sub] = param_fits72[3]
    data_weight72[sub] = param_fits72[4]
    data_gamnonr72[sub] = param_fits72[5]
    data_negLL72[sub] = negLL72
    data_trials72[sub] = trials

    BIC72 = len(bounds7) * np.log(trials) + 2*negLL72
    data_BIC72[sub] = BIC72
    print('model 7.2 done')



#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params


#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params


results72['eps'] = data_eps72
results72[f'alpha'] = data_alpha72
results72[f'unchosen'] = data_unchosen72
results72[f'unchosenseq'] = data_unchosenseq72
results72[f'weight'] = data_weight72
results72[f'gamnonr'] = data_gamnonr72
results72[f'BIC'] = data_BIC72
results72[f'negLL'] = data_negLL72
results72[f'trials'] = data_trials72
results72['Choices'] = choices

results72['sub'] = data_sub


for s in range(len(files)):

    line_to_write = {"sub": data_sub[s], "eps": data_eps72[s], "alpha": data_alpha72[s],"lambda": data_unchosen72[s], "lambda_seq": data_unchosenseq72[s], "w": data_weight72[s], "gamnr": data_gamnonr72[s], "BIC": data_BIC72[s], "negLL": data_negLL72[s], "trials": data_trials72[s], "choices": choices[s]}
    with open("Human_M7_2.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "lambda", "lambda_seq", "w", "gamnr", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()
        