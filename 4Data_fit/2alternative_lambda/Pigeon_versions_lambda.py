#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for pigeons
Pigeon data in seperate files per subject and condition
Model one: eps, alpha, lambda
Model two: eps, alpha
One epsilon, alpha and lambda_unchosen are estimated via a RW model

In this copy, we try multiple alternatives to lambda 
*1: lambda with stricter boundaries (1/K) (as with 1ML_lambda copy.py)
*2: lambda seperate from alpha RW and with weight in decision-policy
*3: lambda seperate from alpha RW but without weight
*4: lambda seperate from alpha RW with its own RW + weight (there is a mistake in the lambda RW rule, a - should be a +)
*5: binary lambda with weight (adding 1/K)
*6: binary lambda without weight (adding 1/K)
*7: binary lambda with weight (subtracting 1/K)
*8: binary lambda without weight (subtracting 1/K)

Only these are mentioned in paper
update after meeting on 2/7/2025:
*9: model 2 but weight is V = Q + wF and w unbounded
*10: model 4 but weight is V = Q + wF and w unbounded
*11: model 5 but weight is V = Q + wF and w unbounded
*12: model 7 but weight is V = Q + wF and w unbounded

"""
import time
start = time.time()
from scipy.optimize import minimize # finding optimal params in models
from scipy import stats             # statistical tools
import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import matplotlib.pyplot as plt     # plotting
from scipy.optimize import differential_evolution # finding optimal params in models
import pickle
import random


#Frist: Read Neuringer's data
data_dir1 = os.path.dirname(os.path.abspath(__file__))
data_dir2 = os.path.join(data_dir1, 'ALL_DATA')
data_dir = os.path.join(data_dir2, 'Pigeon')

dir_sub1con2 = os.path.join(data_dir, 'sub1con2.xlsx')
Sub1_data2 = pd.read_excel(dir_sub1con2)
sub1_data2 = Sub1_data2.to_numpy()

dir_sub1con4 = os.path.join(data_dir, 'sub1con4.xlsx')
Sub1_data4 = pd.read_excel(dir_sub1con4)
sub1_data4 = Sub1_data4.to_numpy() 

dir_sub1con8 = os.path.join(data_dir, 'sub1con8.xlsx')
Sub1_data8 = pd.read_excel(dir_sub1con8)
sub1_data8 = Sub1_data8.to_numpy()


dir_sub2con2 = os.path.join(data_dir, 'sub2con2.xlsx')
Sub2_data2 = pd.read_excel(dir_sub2con2)
sub2_data2 = Sub2_data2.to_numpy()

dir_sub2con4 = os.path.join(data_dir, 'sub2con4.xlsx')
Sub2_data4 = pd.read_excel(dir_sub2con4)
sub2_data4 = Sub2_data4.to_numpy()

dir_sub2con8 = os.path.join(data_dir, 'sub2con8.xlsx')
Sub2_data8 = pd.read_excel(dir_sub2con8)
sub2_data8 = Sub2_data8.to_numpy()



dir_sub3con2 = os.path.join(data_dir, 'sub3con2.xlsx')
Sub3_data2 = pd.read_excel(dir_sub3con2)
sub3_data2 = Sub3_data2.to_numpy()

dir_sub3con4 = os.path.join(data_dir, 'sub3con4.xlsx')
Sub3_data4 = pd.read_excel(dir_sub3con4)
sub3_data4 = Sub3_data4.to_numpy()

dir_sub3con8 = os.path.join(data_dir, 'sub3con8.xlsx')
Sub3_data8 = pd.read_excel(dir_sub3con8)
sub3_data8 = Sub3_data8.to_numpy()



dir_sub4con2 = os.path.join(data_dir, 'sub4con2.xlsx')
Sub4_data2 = pd.read_excel(dir_sub4con2)
sub4_data2 = Sub4_data2.to_numpy()

dir_sub4con4 = os.path.join(data_dir, 'sub4con4.xlsx')
Sub4_data4 = pd.read_excel(dir_sub4con4)
sub4_data4 = Sub4_data4.to_numpy()

dir_sub4con8 = os.path.join(data_dir, 'sub4con8.xlsx')
Sub4_data8 = pd.read_excel(dir_sub4con8)
sub4_data8 = Sub4_data8.to_numpy()



dir_sub5con2 = os.path.join(data_dir, 'sub5con2.xlsx')
Sub5_data2 = pd.read_excel(dir_sub5con2)
sub5_data2 = Sub5_data2.to_numpy()

dir_sub5con4 = os.path.join(data_dir, 'sub5con4.xlsx')
Sub5_data4 = pd.read_excel(dir_sub5con4)
sub5_data4 = Sub5_data4.to_numpy()

dir_sub5con8 = os.path.join(data_dir, 'sub5con8.xlsx')
Sub5_data8 = pd.read_excel(dir_sub5con8)
sub5_data8 = Sub5_data8.to_numpy()


#Second, define the negative log likelihood function
#model 1
#alpha, eps, unchosen = params
def negll_RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen = params
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

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 2
#alpha, eps, unchosen, w = params
def negll_2RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = w*Q_k + (1-w)*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] += unchosen

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 3
#alpha, eps, unchosen = params
def negll_3RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] += unchosen

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model 4
#alpha, eps, unchosen, w = params
def negll_4RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = w*Q_k + (1-w)*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        F_k += F_k + unchosen*(1-F_k)
        F_k[k[t]] = F_k[k[t]] - unchosen*(1-F_k[k[t]])
        F_k[k[t]] = F_k[k[t]] - unchosen*(0-F_k[k[t]])

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL




#model 5
#alpha, eps, w = params
def negll_5RW_eps_alpha_uc(params, k, r):

    alpha, eps, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = w*Q_k + (1-w)*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] += 1/K

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 6
#alpha, eps= params
def negll_6RW_eps_alpha_uc(params, k, r):

    alpha, eps= params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] += 1/K

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL



#model 7
#alpha, eps, w = params
def negll_7RW_eps_alpha_uc(params, k, r):

    alpha, eps, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = w*Q_k + (1-w)*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] -= 1/K

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 8
#alpha, eps= params
def negll_8RW_eps_alpha_uc(params, k, r):

    alpha, eps= params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] -= 1/K

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


    
#model 9
#alpha, eps, unchosen, w = params
def negll_9RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + w*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] += unchosen

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 10
#alpha, eps, unchosen, w = params
def negll_10RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + w*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        F_k += F_k + unchosen*(1-F_k)
        F_k[k[t]] = F_k[k[t]] - unchosen*(1-F_k[k[t]])
        F_k[k[t]] = F_k[k[t]] + unchosen*(0-F_k[k[t]])

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 11
#alpha, eps, w = params
def negll_11RW_eps_alpha_uc(params, k, r):

    alpha, eps, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + w*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] += 1/K

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model 12
#alpha, eps, w = params
def negll_12RW_eps_alpha_uc(params, k, r):

    alpha, eps, w = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    F_k = np.zeros(K) #initual value of Q for each choice
    F_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        F_k_stored[t,:] = F_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        FandQ = Q_k + w*F_k
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(FandQ)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        F_k[np.arange(len(Q_k)) != k[t]] -= 1/K

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL




Q_int = 1






strategy = 'randtobest1bin'
rewards = {}
mean_rewards = np.zeros(15)
standarderror_rewards = np.zeros(15)




results1 = {}
data_eps1 = np.zeros(15, dtype=float)
data_alpha1 = np.zeros(15, dtype=float)
data_unchosen1 = np.zeros(15, dtype=float)
data_BIC1 = np.zeros(15, dtype=float)
data_negLL1 = np.zeros(15, dtype=float)
data_trials1 = np.zeros(15, dtype=float)

results2 = {}
data_eps2 = np.zeros(15, dtype=float)
data_alpha2 = np.zeros(15, dtype=float)
data_unchosen2 = np.zeros(15, dtype=float)
data_weight2 = np.zeros(15, dtype=float)
data_BIC2 = np.zeros(15, dtype=float)
data_negLL2 = np.zeros(15, dtype=float)
data_trials2 = np.zeros(15, dtype=float)

results3 = {}
data_eps3 = np.zeros(15, dtype=float)
data_alpha3 = np.zeros(15, dtype=float)
data_unchosen3 = np.zeros(15, dtype=float)
data_BIC3 = np.zeros(15, dtype=float)
data_negLL3 = np.zeros(15, dtype=float)
data_trials3 = np.zeros(15, dtype=float)


results4 = {}
data_eps4 = np.zeros(15, dtype=float)
data_alpha4 = np.zeros(15, dtype=float)
data_unchosen4 = np.zeros(15, dtype=float)
data_weight4 = np.zeros(15, dtype=float)
data_BIC4 = np.zeros(15, dtype=float)
data_negLL4 = np.zeros(15, dtype=float)
data_trials4 = np.zeros(15, dtype=float)





results5 = {}
data_eps5 = np.zeros(15, dtype=float)
data_alpha5 = np.zeros(15, dtype=float)
data_weight5 = np.zeros(15, dtype=float)
data_BIC5 = np.zeros(15, dtype=float)
data_negLL5 = np.zeros(15, dtype=float)
data_trials5 = np.zeros(15, dtype=float)


results6 = {}
data_eps6 = np.zeros(15, dtype=float)
data_alpha6 = np.zeros(15, dtype=float)
data_BIC6 = np.zeros(15, dtype=float)
data_negLL6 = np.zeros(15, dtype=float)
data_trials6 = np.zeros(15, dtype=float)


results7 = {}
data_eps7 = np.zeros(15, dtype=float)
data_alpha7 = np.zeros(15, dtype=float)
data_weight7 = np.zeros(15, dtype=float)
data_BIC7 = np.zeros(15, dtype=float)
data_negLL7 = np.zeros(15, dtype=float)
data_trials7 = np.zeros(15, dtype=float)


results8 = {}
data_eps8 = np.zeros(15, dtype=float)
data_alpha8 = np.zeros(15, dtype=float)
data_BIC8 = np.zeros(15, dtype=float)
data_negLL8 = np.zeros(15, dtype=float)
data_trials8 = np.zeros(15, dtype=float)




results9 = {}
data_eps9 = np.zeros(15, dtype=float)
data_alpha9 = np.zeros(15, dtype=float)
data_unchosen9 = np.zeros(15, dtype=float)
data_weight9 = np.zeros(15, dtype=float)
data_BIC9 = np.zeros(15, dtype=float)
data_negLL9 = np.zeros(15, dtype=float)
data_trials9 = np.zeros(15, dtype=float)

results10 = {}
data_eps10 = np.zeros(15, dtype=float)
data_alpha10 = np.zeros(15, dtype=float)
data_unchosen10 = np.zeros(15, dtype=float)
data_weight10 = np.zeros(15, dtype=float)
data_BIC10 = np.zeros(15, dtype=float)
data_negLL10 = np.zeros(15, dtype=float)
data_trials10 = np.zeros(15, dtype=float)


results11 = {}
data_eps11 = np.zeros(15, dtype=float)
data_alpha11 = np.zeros(15, dtype=float)
data_unchosen11 = np.zeros(15, dtype=float)
data_weight11 = np.zeros(15, dtype=float)
data_BIC11 = np.zeros(15, dtype=float)
data_negLL11 = np.zeros(15, dtype=float)
data_trials11 = np.zeros(15, dtype=float)

results12 = {}
data_eps12 = np.zeros(15, dtype=float)
data_alpha12 = np.zeros(15, dtype=float)
data_weight12 = np.zeros(15, dtype=float)
data_BIC12 = np.zeros(15, dtype=float)
data_negLL12 = np.zeros(15, dtype=float)
data_trials12 = np.zeros(15, dtype=float)



choices = np.zeros(15,dtype=float)



for sub, sub_data in enumerate([sub1_data2, sub2_data2, sub3_data2, sub4_data2, sub5_data2, sub1_data4, sub2_data4, sub3_data4, sub4_data4, sub5_data4, sub1_data8, sub2_data8, sub3_data8, sub4_data8, sub5_data8]):


    print('starting at sub', {sub})
    trials = np.shape(sub_data)[0]
    


    k= sub_data[:,4]
    k= np.add(k,-1)
    k= k.astype(int)
            
    r= sub_data[:,5]
    r= r.astype(int)

    K = np.max(k)+1
    print('K is', K)

    


    bounds1=[(0,1), (0,1), (-1/K,1/K)]
    bounds2=[(0,1), (0,1), (-1/K,1/K), (0,1)]
    bounds3=[(0,1), (0,1), (-1/K,1/K)]
    bounds4=[(0,1), (0,1), (-1/K,1/K), (0,1)]


    bounds5=[(0,1), (0,1), (0,1)]
    bounds6=[(0,1), (0,1)]
    bounds7=[(0,1), (0,1), (0,1)]
    bounds8=[(0,1), (0,1)]

    


    bounds9=[(0,1), (0,1), (-1/K,1/K), (-10,10)]
    bounds10=[(0,1), (0,1), (-1/K,1/K), (-10,10)]
    bounds11=[(0,1), (0,1), (-10,10)]
    bounds12=[(0,1), (0,1), (-10,10)]


    choices[sub] = sub_data[10,1]

    seedings = int(np.sum(k[5:10])+np.sum(r[16:52])+(sub+1))
    print('random seed for', sub, 'is', seedings)

    random.seed(seedings)
    np.random.seed(seedings)

    
    '''

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result1 = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL1 = result1.fun
    param_fits1 = result1.x

    data_alpha1[sub] = param_fits1[0]
    data_eps1[sub] = param_fits1[1]
    data_unchosen1[sub] = param_fits1[2]
    data_negLL1[sub] = negLL1
    data_trials1[sub] = trials

    BIC1 = len(bounds1) * np.log(trials) + 2*negLL1
    data_BIC1[sub] = BIC1
    print('model 1 done')


   #fit model 2
    result2 = differential_evolution(negll_2RW_eps_alpha_uc, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL2 = result2.fun
    param_fits2 = result2.x

    data_alpha2[sub] = param_fits2[0]
    data_eps2[sub] = param_fits2[1]
    data_unchosen2[sub] = param_fits2[2]
    data_weight2[sub] = param_fits2[3]
    data_negLL2[sub] = negLL2
    data_trials2[sub] = trials

    BIC2 = len(bounds2) * np.log(trials) + 2*negLL2
    data_BIC2[sub] = BIC2
    print('model 2 done')



    #fit model 3
    result3 = differential_evolution(negll_3RW_eps_alpha_uc, bounds=bounds3, args=(k,r), strategy=strategy)
    negLL3 = result3.fun
    param_fits3 = result3.x

    data_alpha3[sub] = param_fits3[0]
    data_eps3[sub] = param_fits3[1]
    data_unchosen3[sub] = param_fits3[2]
    data_negLL3[sub] = negLL3
    data_trials3[sub] = trials

    BIC3 = len(bounds3) * np.log(trials) + 2*negLL3
    data_BIC3[sub] = BIC3
    print('model 3 done')


    #fit model 4
    result4 = differential_evolution(negll_4RW_eps_alpha_uc, bounds=bounds4, args=(k,r), strategy=strategy)
    negLL4 = result4.fun
    param_fits4 = result4.x

    data_alpha4[sub] = param_fits4[0]
    data_eps4[sub] = param_fits4[1]
    data_unchosen4[sub] = param_fits4[2]
    data_weight4[sub] = param_fits4[3]
    data_negLL4[sub] = negLL4
    data_trials4[sub] = trials

    BIC4 = len(bounds4) * np.log(trials) + 2*negLL4
    data_BIC4[sub] = BIC4
    print('model 4 done')




    #fit model 5
    result5 = differential_evolution(negll_5RW_eps_alpha_uc, bounds=bounds5, args=(k,r), strategy=strategy)
    negLL5 = result5.fun
    param_fits5 = result5.x

    data_alpha5[sub] = param_fits5[0]
    data_eps5[sub] = param_fits5[1]
    data_weight5[sub] = param_fits5[2]
    data_negLL5[sub] = negLL5
    data_trials5[sub] = trials

    BIC5 = len(bounds5) * np.log(trials) + 2*negLL5
    data_BIC5[sub] = BIC5
    print('model 5 done')


    #fit model 6
    result6 = differential_evolution(negll_6RW_eps_alpha_uc, bounds=bounds6, args=(k,r), strategy=strategy)
    negLL6 = result6.fun
    param_fits6 = result6.x

    data_alpha6[sub] = param_fits6[0]
    data_eps6[sub] = param_fits6[1]
    data_negLL6[sub] = negLL6
    data_trials6[sub] = trials

    BIC6 = len(bounds6) * np.log(trials) + 2*negLL6
    data_BIC6[sub] = BIC6
    print('model 6 done')


    #fit model 7
    result7 = differential_evolution(negll_7RW_eps_alpha_uc, bounds=bounds7, args=(k,r), strategy=strategy)
    negLL7 = result7.fun
    param_fits7 = result7.x

    data_alpha7[sub] = param_fits7[0]
    data_eps7[sub] = param_fits7[1]
    data_weight7[sub] = param_fits7[2]
    data_negLL7[sub] = negLL7
    data_trials7[sub] = trials

    BIC7 = len(bounds7) * np.log(trials) + 2*negLL7
    data_BIC7[sub] = BIC7
    print('model 7 done')


    #fit model 8
    result8 = differential_evolution(negll_8RW_eps_alpha_uc, bounds=bounds8, args=(k,r), strategy=strategy)
    negLL8 = result8.fun
    param_fits8 = result8.x

    data_alpha8[sub] = param_fits8[0]
    data_eps8[sub] = param_fits7[1]
    data_negLL8[sub] = negLL8
    data_trials8[sub] = trials

    BIC8 = len(bounds8) * np.log(trials) + 2*negLL8
    data_BIC8[sub] = BIC8
    print('model 8 done')

    

    #fit model 9
    result9 = differential_evolution(negll_9RW_eps_alpha_uc, bounds=bounds9, args=(k,r), strategy=strategy)
    negLL9 = result9.fun
    param_fits9 = result9.x

    data_alpha9[sub] = param_fits9[0]
    data_eps9[sub] = param_fits9[1]
    data_unchosen9[sub] = param_fits9[2]
    data_weight9[sub] = param_fits9[3]
    data_negLL9[sub] = negLL9
    data_trials9[sub] = trials

    BIC9 = len(bounds9) * np.log(trials) + 2*negLL9
    data_BIC9[sub] = BIC9
    print('model 9 done')

    '''
    #fit model 10
    result10 = differential_evolution(negll_10RW_eps_alpha_uc, bounds=bounds10, args=(k,r), strategy=strategy)
    negLL10 = result10.fun
    param_fits10 = result10.x

    data_alpha10[sub] = param_fits10[0]
    data_eps10[sub] = param_fits10[1]
    data_unchosen10[sub] = param_fits10[2]
    data_weight10[sub] = param_fits10[3]
    data_negLL10[sub] = negLL10
    data_trials10[sub] = trials

    BIC10 = len(bounds10) * np.log(trials) + 2*negLL10
    data_BIC10[sub] = BIC10
    print('model 10 done')
    print('pigoen sub', sub, 'BIC', BIC10)

    


    '''

    #fit model 11
    result11 = differential_evolution(negll_11RW_eps_alpha_uc, bounds=bounds11, args=(k,r), strategy=strategy)
    negLL11 = result11.fun
    param_fits11 = result11.x

    data_alpha11[sub] = param_fits11[0]
    data_eps11[sub] = param_fits11[1]
    data_weight11[sub] = param_fits11[2]
    data_negLL11[sub] = negLL11
    data_trials11[sub] = trials

    BIC11 = len(bounds11) * np.log(trials) + 2*negLL11
    data_BIC11[sub] = BIC11
    print('model 11 done')



    #fit model 12
    result12 = differential_evolution(negll_12RW_eps_alpha_uc, bounds=bounds12, args=(k,r), strategy=strategy)
    negLL12 = result12.fun
    param_fits12 = result12.x

    data_alpha12[sub] = param_fits12[0]
    data_eps12[sub] = param_fits12[1]
    data_weight12[sub] = param_fits12[2]
    data_negLL12[sub] = negLL12
    data_trials12[sub] = trials

    BIC12 = len(bounds12) * np.log(trials) + 2*negLL12
    data_BIC12[sub] = BIC12
    print('model 12 done')







results1['eps'] = data_eps1
results1[f'alpha'] = data_alpha1
results1[f'unchosen'] = data_unchosen1
results1[f'BIC'] = data_BIC1
results1[f'negLL'] = data_negLL1
results1[f'trials'] = data_trials1
results1['Choices'] = choices
    
results2['eps'] = data_eps2
results2[f'alpha'] = data_alpha2
results2[f'unchosen'] = data_unchosen2
results2[f'weight'] = data_weight2
results2[f'BIC'] = data_BIC2
results2[f'negLL'] = data_negLL2
results2[f'trials'] = data_trials2
results2['Choices'] = choices

results3['eps'] = data_eps3
results3[f'alpha'] = data_alpha3
results3[f'unchosen'] = data_unchosen3
results3[f'BIC'] = data_BIC3
results3[f'negLL'] = data_negLL3
results3[f'trials'] = data_trials3
results3['Choices'] = choices
    
results4['eps'] = data_eps4
results4[f'alpha'] = data_alpha4
results4[f'unchosen'] = data_unchosen4
results4[f'weight'] = data_weight4
results4[f'BIC'] = data_BIC4
results4[f'negLL'] = data_negLL4
results4[f'trials'] = data_trials4
results4['Choices'] = choices





results5['eps'] = data_eps5
results5[f'alpha'] = data_alpha5
results5[f'weight'] = data_weight5
results5[f'BIC'] = data_BIC5
results5[f'negLL'] = data_negLL5
results5[f'trials'] = data_trials5
results5['Choices'] = choices
    
results6['eps'] = data_eps6
results6[f'alpha'] = data_alpha6
results6[f'BIC'] = data_BIC6
results6[f'negLL'] = data_negLL6
results6[f'trials'] = data_trials6
results6['Choices'] = choices

results7['eps'] = data_eps7
results7[f'alpha'] = data_alpha7
results7[f'weight'] = data_weight7
results7[f'BIC'] = data_BIC7
results7[f'negLL'] = data_negLL7
results7[f'trials'] = data_trials7
results7['Choices'] = choices
    
results8['eps'] = data_eps8
results8[f'alpha'] = data_alpha8
results8[f'BIC'] = data_BIC8
results8[f'negLL'] = data_negLL8
results8[f'trials'] = data_trials8
results8['Choices'] = choices





results9['eps'] = data_eps9
results9[f'alpha'] = data_alpha9
results9[f'unchosen'] = data_unchosen9
results9[f'weight'] = data_weight9
results9[f'BIC'] = data_BIC9
results9[f'negLL'] = data_negLL9
results9[f'trials'] = data_trials9
results9['Choices'] = choices

'''
results10['eps'] = data_eps10
results10[f'alpha'] = data_alpha10
results10[f'unchosen'] = data_unchosen10
results10[f'weight'] = data_weight10
results10[f'BIC'] = data_BIC10
results10[f'negLL'] = data_negLL10
results10[f'trials'] = data_trials10
results10['Choices'] = choices

'''
results11['eps'] = data_eps11
results11[f'alpha'] = data_alpha11
results11[f'weight'] = data_weight11
results11[f'BIC'] = data_BIC11
results11[f'negLL'] = data_negLL11
results11[f'trials'] = data_trials11
results11['Choices'] = choices

results12['eps'] = data_eps12
results12[f'alpha'] = data_alpha12
results12[f'weight'] = data_weight12
results12[f'BIC'] = data_BIC12
results12[f'negLL'] = data_negLL12
results12[f'trials'] = data_trials12
results12['Choices'] = choices

BIC = {}

BIC['eps-alpha-uc'] = data_BIC1
BIC['seperate-uc-weight'] = data_BIC2
BIC['seperate-uc-noweight'] = data_BIC3
BIC['RWuc-weight'] = data_BIC4
BIC['eps-alpha-posuc-weight'] = data_BIC5
BIC['eps-alpha-posuc'] = data_BIC6
BIC['eps-alpha-neguc-weight'] = data_BIC7
BIC['eps-alpha-neguc'] = data_BIC8

BIC['seperate-uc-difweight'] = data_BIC9
BIC['RWuc-difweight'] = data_BIC10
BIC['eps-alpha-posuc-difweight'] = data_BIC11
BIC['eps-alpha-neguc-difweight'] = data_BIC12
'''

save_dir = os.path.join(data_dir1, 'Pigeon_output_versions_lambda')
'''

dict_name = f'1Pigeon_eps-alpha-uc.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results1)
df.to_excel(dict_save)


dict_name = f'2Pigeon_seperate_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results2)
df.to_excel(dict_save)

dict_name = f'3Pigeon_seperate_uc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results3)
df.to_excel(dict_save)

dict_name = f'4Pigeon_RW_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results4)
df.to_excel(dict_save)




dict_name = f'5Pigeon_eps-alpha-posuc-weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results5)
df.to_excel(dict_save)

dict_name = f'6Pigeon_eps-alpha-posuc.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results6)
df.to_excel(dict_save)

dict_name = f'7Pigeon_eps-alpha-neguc-weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results7)
df.to_excel(dict_save)

dict_name = f'8Pigeon_eps-alpha-neguc.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results8)
df.to_excel(dict_save)



dict_name = f'9Pigeon_seperate_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results9)
df.to_excel(dict_save)


'''
dict_name = f'10Pigeon_RW_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results10)
df.to_excel(dict_save)
'''



dict_name = f'11Pigeon_eps-alpha-posuc-weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results11)
df.to_excel(dict_save)


dict_name = f'12Pigeon_eps-alpha-neguc-weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results12)
df.to_excel(dict_save)


dict_name = f'Pigeon_BIC.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=BIC)
df.to_excel(dict_save)


'''

print(data_BIC10)