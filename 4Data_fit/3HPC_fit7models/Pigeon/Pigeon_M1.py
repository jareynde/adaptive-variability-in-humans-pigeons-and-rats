#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for pigeons
Pigeon data in seperate files per subject and condition

In this copy, we fit these models:
*1: eps-lr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model
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

#model 2
def negll_RW_eps_lr_uc(params, k, r):

    eps, lr, uc = params
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

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

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


#model 4.1
def negll_RW_eps_lr_uc_gamr(params, k, r, buffer):

    eps, lr, uc, gamr = params

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

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced



    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

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



#model 7.1
def negll_RW_eps_lr_uc_ucs_gamr(params, k, r, buffer):

    eps, lr, uc, ucs, w, gamr = params
    

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
mean_rewards = np.zeros(15)
standarderror_rewards = np.zeros(15)

data_sub = np.zeros(15,dtype=float)

#    model 1    eps, lr = params
results1 = {}
data_eps1 = np.zeros(15, dtype=float)
data_alpha1 = np.zeros(15, dtype=float)
data_BIC1 = np.zeros(15, dtype=float)
data_negLL1 = np.zeros(15, dtype=float)
data_trials1 = np.zeros(15, dtype=float)


#    model 2    eps, lr, uc = params
results2 = {}
data_eps2 = np.zeros(15, dtype=float)
data_alpha2 = np.zeros(15, dtype=float)
data_unchosen2 = np.zeros(15, dtype=float)
data_BIC2 = np.zeros(15, dtype=float)
data_negLL2 = np.zeros(15, dtype=float)
data_trials2 = np.zeros(15, dtype=float)

#    model 3.1  eps, lr, gamr = params
results31 = {}
data_eps31 = np.zeros(15, dtype=float)
data_alpha31 = np.zeros(15, dtype=float)
data_gamr31 = np.zeros(15, dtype=float)
data_BIC31 = np.zeros(15, dtype=float)
data_negLL31 = np.zeros(15, dtype=float)
data_trials31 = np.zeros(15, dtype=float)


#    model 3.2  eps, lr, gamnr = params
results32 = {}
data_eps32 = np.zeros(15, dtype=float)
data_alpha32 = np.zeros(15, dtype=float)
data_gamnonr32 = np.zeros(15, dtype=float)
data_BIC32 = np.zeros(15, dtype=float)
data_negLL32 = np.zeros(15, dtype=float)
data_trials32 = np.zeros(15, dtype=float)



#    model 4.1  eps, lr, uc, gamr = params
results41 = {}
data_eps41 = np.zeros(15, dtype=float)
data_alpha41 = np.zeros(15, dtype=float)
data_unchosen41 = np.zeros(15, dtype=float)
data_gamr41 = np.zeros(15, dtype=float)
data_BIC41 = np.zeros(15, dtype=float)
data_negLL41 = np.zeros(15, dtype=float)
data_trials41 = np.zeros(15, dtype=float)

#   model 4.2  eps, lr, uc, gamnr = params
results42 = {}
data_eps42 = np.zeros(15, dtype=float)
data_alpha42 = np.zeros(15, dtype=float)
data_unchosen42 = np.zeros(15, dtype=float)
data_gamnonr42 = np.zeros(15, dtype=float)
data_BIC42 = np.zeros(15, dtype=float)
data_negLL42 = np.zeros(15, dtype=float)
data_trials42 = np.zeros(15, dtype=float)



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




#    model 7.1  eps, lr, uc, ucs, w, gamr = params
results71 = {}
data_eps71 = np.zeros(15, dtype=float)
data_alpha71 = np.zeros(15, dtype=float)
data_unchosen71 = np.zeros(15, dtype=float)
data_unchosenseq71 = np.zeros(15,dtype=float)
data_weight71 = np.zeros(15, dtype=float)
data_gamr71 = np.zeros(15, dtype=float)
data_BIC71 = np.zeros(15, dtype=float)
data_negLL71 = np.zeros(15, dtype=float)
data_trials71 = np.zeros(15, dtype=float)


#    model 7.2  eps, lr, uc, ucs, w, gamnr = params
results72= {}
data_eps72 = np.zeros(15, dtype=float)
data_alpha72 = np.zeros(15, dtype=float)
data_unchosen72 = np.zeros(15, dtype=float)
data_unchosenseq72 = np.zeros(15,dtype=float)
data_weight72 = np.zeros(15, dtype=float)
data_gamnonr72 = np.zeros(15, dtype=float)
data_BIC72 = np.zeros(15, dtype=float)
data_negLL72 = np.zeros(15, dtype=float)
data_trials72 = np.zeros(15, dtype=float)




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

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    #eps, lr = params
    result1 = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL1 = result1.fun
    param_fits1 = result1.x
    
    data_eps1[sub] = param_fits1[0]
    data_alpha1[sub] = param_fits1[1]
    data_negLL1[sub] = negLL1
    data_trials1[sub] = trials

    BIC1 = len(bounds1) * np.log(trials) + 2*negLL1
    data_BIC1[sub] = BIC1
    print('model 1 done')

"""

    #fit model 2
    #eps, lr, uc = params
    result2 = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL2 = result2.fun
    param_fits2 = result2.x

    data_eps2[sub] = param_fits2[0]
    data_alpha2[sub] = param_fits2[1]
    data_unchosen2[sub] = param_fits2[2]
    data_negLL2[sub] = negLL2
    data_trials2[sub] = trials

    BIC2 = len(bounds2) * np.log(trials) + 2*negLL2
    data_BIC2[sub] = BIC2
    print('model 2 done')



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


    #fit model 4.1
    #eps, lr, uc, gamr = params
    result41 = differential_evolution(negll_RW_eps_lr_uc_gamr, bounds=bounds4, args=(k,r,buffer), strategy=strategy, constraints=constraint4)
    negLL41 = result41.fun
    param_fits41 = result41.x

    data_eps41[sub] = param_fits41[0]
    data_alpha41[sub] = param_fits41[1]
    data_unchosen41[sub] = param_fits41[2]
    data_gamr41[sub] = param_fits41[3]
    data_negLL41[sub] = negLL41
    data_trials41[sub] = trials

    BIC41 = len(bounds4) * np.log(trials) + 2*negLL41
    data_BIC41[sub] = BIC41
    print('model 4.1 done')


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


   

    #fit model 7.1
    #eps, lr, uc, ucs, w, gamr = params
    result71 = differential_evolution(negll_RW_eps_lr_uc_ucs_gamr, bounds=bounds7, args=(k,r,buffer), strategy=strategy, constraints=constraint7)
    negLL71 = result71.fun
    param_fits71 = result71.x

    data_eps71[sub] = param_fits71[0]
    data_alpha71[sub] = param_fits71[1]
    data_unchosen71[sub] = param_fits71[2]
    data_unchosenseq71[sub] = param_fits71[3]
    data_weight71[sub] = param_fits71[4]
    data_gamr71[sub] = param_fits71[5]
    data_negLL71[sub] = negLL71
    data_trials71[sub] = trials

    BIC71 = len(bounds7) * np.log(trials) + 2*negLL71
    data_BIC71[sub] = BIC71
    print('model 7.1 done')


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


"""

#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params


#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params



'''


results2['eps'] = data_eps2
results2[f'alpha'] = data_alpha2
results2[f'unchosen'] = data_unchosen2
results2[f'BIC'] = data_BIC2
results2[f'negLL'] = data_negLL2
results2[f'trials'] = data_trials2
results2['Choices'] = choices

results31['eps'] = data_eps31
results31[f'alpha'] = data_alpha31
results31[f'gamr'] = data_gamr31
results31[f'BIC'] = data_BIC31
results31[f'negLL'] = data_negLL31
results31[f'trials'] = data_trials31
results31['Choices'] = choices

results32['eps'] = data_eps32
results32[f'alpha'] = data_alpha32
results32[f'gamnonr'] = data_gamnonr32
results32[f'BIC'] = data_BIC32
results32[f'negLL'] = data_negLL32
results32[f'trials'] = data_trials32
results32['Choices'] = choices
    
results41['eps'] = data_eps41
results41[f'alpha'] = data_alpha41
results41[f'unchosen'] = data_unchosen41
results41[f'gamr'] = data_gamr41
results41[f'BIC'] = data_BIC41
results41[f'negLL'] = data_negLL41
results41[f'trials'] = data_trials41
results41['Choices'] = choices


results42['eps'] = data_eps42
results42[f'alpha'] = data_alpha42
results42[f'unchosen'] = data_unchosen42
results42[f'gamnonr'] = data_gamnonr42
results42[f'BIC'] = data_BIC42
results42[f'negLL'] = data_negLL42
results42[f'trials'] = data_trials42
results42['Choices'] = choices



results5['eps'] = data_eps5
results5[f'alpha'] = data_alpha5
results5[f'unchosen'] = data_unchosen5
results5[f'unchosenseq'] = data_unchosenseq5
results5[f'weight'] = data_weight5
results5[f'BIC'] = data_BIC5
results5[f'negLL'] = data_negLL5
results5[f'trials'] = data_trials5
results5['Choices'] = choices
    


results71['eps'] = data_eps71
results71[f'alpha'] = data_alpha71
results71[f'unchosen'] = data_unchosen71
results71[f'unchosenseq'] = data_unchosenseq71
results71[f'weight'] = data_weight71
results71[f'gamr'] = data_gamr71
results71[f'BIC'] = data_BIC71
results71[f'negLL'] = data_negLL71
results71[f'trials'] = data_trials71
results71['Choices'] = choices

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

'''

results1['sub'] = data_sub
results2['sub'] = data_sub
results31['sub'] = data_sub
results32['sub'] = data_sub
results41['sub'] = data_sub
results42['sub'] = data_sub
results5['sub'] = data_sub
results71['sub'] = data_sub
results72['sub'] = data_sub


results1['eps'] = data_eps1
results1[f'alpha'] = data_alpha1
results1[f'BIC'] = data_BIC1
results1[f'negLL'] = data_negLL1
results1[f'trials'] = data_trials1
results1['Choices'] = choices

for s in range(len(files)):

    line_to_write = {"sub": data_sub[s], "eps": data_eps1[s], "alpha": data_alpha1[s],"BIC": data_BIC1[s], "negLL": data_negLL1[s], "trials": data_trials1[s], "choices": choices[s]}
    with open("Pigeon_M1.csv", 'a') as f_object:
        field_names = ["sub", "eps", "alpha", "BIC", "negLL", "trials", "choices"]
        dictwriter_object = DictWriter(f_object, fieldnames = field_names)
        dictwriter_object.writerow(line_to_write)
        f_object.close()