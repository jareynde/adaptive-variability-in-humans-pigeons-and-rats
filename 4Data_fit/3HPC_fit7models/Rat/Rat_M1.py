#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for rats

In this copy, we fit these models:
*1: eps-lr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#model 1
def negll_RW_eps_lr(params, k, r):

    eps, lr = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

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


species_folder = f"AnimalModelFit/ALL_DATA_CSV/Rats"
subj_folders = os.listdir(species_folder)


for count, folder in enumerate(subj_folders):
    count=count+1
    subj_dir = os.path.join(species_folder,f'{folder}')

    subj_data_list = os.listdir(subj_dir)

    sub_eps = np.array([]) 
    sub_alpha = np.array([]) 
    sub_BIC = np.array([]) 
    sub_negLL = np.array([]) 
    sub_trials = np.array([])

    for file_name in subj_data_list:
        data_dir = os.path.join(subj_dir, file_name)
        Data = pd.read_csv(data_dir)
        data = Data.to_numpy()
        
        trials = np.shape(data)[0]

        k= data[:,1]
        k= np.add(k,-1)
        k= k.astype(int)
                
        r= data[:,3]
        r= r.astype(int)

        K = np.max(k)+1

        bounds1=[(0,1), (0,1)]
        bounds2=[(0,1), (0,1),(-1/K,1/K)]
        bounds3=[(0,1), (0,1), (0,1)]
        bounds4=[(0,1), (0,1), (-1/K,1/K), (0,1)]
        bounds5=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,1)]
        bounds7=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,1), (0,1)]

        buffer = 3
        

#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params

#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params


        #fit model 1: RW, fitting eps, alpha and lambda unchosen
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(trials) + 2*negLL

        
        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps = np.append(sub_eps, param_fits[0])
        sub_alpha = np.append(sub_alpha, param_fits[1])
        sub_BIC = np.append(sub_BIC, BIC)
        sub_negLL = np.append(sub_negLL, negLL)
        sub_trials = np.append(sub_trials, trials)

        line_to_write = {"sub": count, "file": file_name, "eps": param_fits[0], "alpha": param_fits[1], "BIC": BIC, "negLL": negLL, "trials": trials, "choices": K}
        with open("Ratall_M1.csv", 'a') as f_object:
            field_names = ["sub", "file", "eps", "alpha", "BIC", "negLL", "trials", "choices"]
            dictwriter_object = DictWriter(f_object, fieldnames = field_names)
            dictwriter_object.writerow(line_to_write)
            f_object.close()


    sub_av_eps = np.mean(sub_eps)
    sub_av_alpha = np.mean(sub_alpha)
    sub_av_BIC = np.mean(sub_BIC)
    sub_av_negLL = np.mean(sub_negLL)
    sub_av_trials = np.mean(sub_trials)

    line_to_write = {"sub": count, "eps": sub_av_eps, "alpha": sub_av_alpha, "BIC": sub_av_BIC, "negLL": sub_av_negLL, "trials": sub_av_trials, "choices": K}
    with open("Ratav_M1.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()


