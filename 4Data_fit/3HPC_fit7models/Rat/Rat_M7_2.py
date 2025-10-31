#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for rats

In this copy, we fit these models:
*7.2: eps-lr-uc-ucs-gamnr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


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


species_folder = f"AnimalModelFit/ALL_DATA_CSV/Rats"
subj_folders = os.listdir(species_folder)


for count, folder in enumerate(subj_folders):
    count=count+1
    subj_dir = os.path.join(species_folder,f'{folder}')

    subj_data_list = os.listdir(subj_dir)

    sub_eps = np.array([]) 
    sub_alpha = np.array([]) 
    sub_uc = np.array([]) 
    sub_ucs = np.array([])
    sub_w = np.array([])
    sub_gamnr = np.array([])
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
        bounds5=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,10)]
        bounds7=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,10), (0,1)]

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
        result = differential_evolution(negll_RW_eps_lr_uc_ucs_gamnr, bounds=bounds7, args=(k,r,buffer), strategy=strategy, constraints=constraint7)
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(trials) + 2*negLL

        
        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps = np.append(sub_eps, param_fits[0])
        sub_alpha = np.append(sub_alpha, param_fits[1])
        sub_uc = np.append(sub_uc, param_fits[2])
        sub_ucs = np.append(sub_ucs, param_fits[3])
        sub_w = np.append(sub_w, param_fits[4])
        sub_gamnr = np.append(sub_gamnr, param_fits[5])
        sub_BIC = np.append(sub_BIC, BIC)
        sub_negLL = np.append(sub_negLL, negLL)
        sub_trials = np.append(sub_trials, trials)

        line_to_write = {"sub": count, "file": file_name, "eps": param_fits[0], "alpha": param_fits[1], "lambda": param_fits[2], "lambda_s" : param_fits[3], "w": param_fits[4], "gamnr": param_fits[5], "BIC": BIC, "negLL": negLL, "trials": trials, "choices": K}
        with open("Ratall_M7_2.csv", 'a') as f_object:
            field_names = ["sub", "file", "eps", "alpha", "lambda", "lambda_s", "w", "gamnr", "BIC", "negLL", "trials", "choices"]
            dictwriter_object = DictWriter(f_object, fieldnames = field_names)
            dictwriter_object.writerow(line_to_write)
            f_object.close()


    sub_av_eps = np.mean(sub_eps)
    sub_av_alpha = np.mean(sub_alpha)
    sub_av_uc = np.mean(sub_uc)
    sub_av_ucs = np.mean(sub_ucs)
    sub_av_w = np.mean(sub_w)
    sub_av_gamnr = np.mean(sub_gamnr)
    sub_av_BIC = np.mean(sub_BIC)
    sub_av_negLL = np.mean(sub_negLL)
    sub_av_trials = np.mean(sub_trials)

    line_to_write = {"sub": count, "eps": sub_av_eps, "alpha": sub_av_alpha, "lambda": sub_av_uc, "lambda_s": sub_av_ucs, "w": sub_av_w, "gamnr": sub_av_gamnr, "BIC": sub_av_BIC, "negLL": sub_av_negLL, "trials": sub_av_trials, "choices": K}
    with open("Ratav_M7_2.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "lambda", "lambda_s", "w", "gamnr", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()


