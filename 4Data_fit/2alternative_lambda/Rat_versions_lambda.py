#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Model one: eps, alpha lambda
Model two: eps, alpha
One epsilon, alpha and lambda_unchosen are estimated via a RW model

In this copy, we try multiple alternatives to lambda 
*1: lambda with stricter boundaries (1/K) (as with 1ML_lambda copy.py)
*2: lambda seperate from alpha RW and with weight in decision-policy (V = w*Q + (1-w)*F)
*3: lambda seperate from alpha RW but without weight
*4: lambda seperate from alpha RW with its own RW + weight (V = w*Q + (1-w)*F) (there is a mistake in the lambda RW rule, a - should be a +)
*5: binary lambda with weight (adding 1/K) (V = w*Q + (1-w)*F)
*6: binary lambda without weight (adding 1/K)
*7: binary lambda with weight (subtracting 1/K) (V = w*Q + (1-w)*F)
*8: binary lambda without weight (subtracting 1/K)

Only these are mentioned in paper
update after meeting on 2/7/2025
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





data_dir1 = os.path.dirname(os.path.abspath(__file__))
data_dir2 = os.path.join(data_dir1, 'ALL_DATA')
data_dir3 = os.path.join(data_dir2, 'rats')

file_dir = os.path.join(data_dir3,'V4_usable')

maps = os.listdir(file_dir)

Q_int = 1


strategy = 'randtobest1bin'


rat_nr = np.array([])
file_nr = np.array([])


all_eps1 = np.array([]) 
all_alpha1 = np.array([]) 
all_unchosen1 = np.array([]) 
all_BIC1 = np.array([]) 
all_negLL1 = np.array([]) 
all_trials1 = np.array([]) 

av_eps1 = np.array([]) 
av_alpha1 = np.array([]) 
av_unchosen1 = np.array([]) 
av_BIC1 = np.array([]) 
av_negLL1 = np.array([]) 
av_trials1 = np.array([])
results1 = {}


all_eps2 = np.array([]) 
all_alpha2 = np.array([]) 
all_unchosen2 = np.array([]) 
all_weight2 = np.array([])
all_BIC2= np.array([]) 
all_negLL2 = np.array([]) 
all_trials2 = np.array([]) 

av_eps2 = np.array([]) 
av_alpha2 = np.array([]) 
av_unchosen2 = np.array([]) 
av_weight2 = np.array([])
av_BIC2 = np.array([]) 
av_negLL2 = np.array([]) 
av_trials2 = np.array([])
results2 = {}



all_eps3 = np.array([]) 
all_alpha3 = np.array([]) 
all_unchosen3 = np.array([]) 
all_BIC3 = np.array([]) 
all_negLL3 = np.array([]) 
all_trials3 = np.array([]) 

av_eps3 = np.array([]) 
av_alpha3 = np.array([]) 
av_unchosen3 = np.array([]) 
av_BIC3 = np.array([]) 
av_negLL3 = np.array([]) 
av_trials3 = np.array([])
results3 = {}

all_eps4 = np.array([]) 
all_alpha4 = np.array([]) 
all_unchosen4 = np.array([]) 
all_weight4 = np.array([])
all_BIC4= np.array([]) 
all_negLL4 = np.array([]) 
all_trials4 = np.array([]) 

av_eps4 = np.array([]) 
av_alpha4 = np.array([]) 
av_unchosen4 = np.array([]) 
av_weight4 = np.array([])
av_BIC4 = np.array([]) 
av_negLL4 = np.array([]) 
av_trials4 = np.array([])
results4 = {}




all_eps5 = np.array([]) 
all_alpha5 = np.array([]) 
all_weight5 = np.array([])
all_BIC5 = np.array([]) 
all_negLL5 = np.array([]) 
all_trials5 = np.array([]) 

av_eps5 = np.array([]) 
av_alpha5 = np.array([]) 
av_weight5 = np.array([])
av_BIC5 = np.array([]) 
av_negLL5 = np.array([]) 
av_trials5 = np.array([])
results5 = {}

all_eps6 = np.array([]) 
all_alpha6 = np.array([]) 
all_BIC6= np.array([]) 
all_negLL6 = np.array([]) 
all_trials6 = np.array([]) 

av_eps6 = np.array([]) 
av_alpha6 = np.array([]) 
av_BIC6 = np.array([]) 
av_negLL6 = np.array([]) 
av_trials6 = np.array([])
results6 = {}


all_eps7 = np.array([]) 
all_alpha7 = np.array([]) 
all_weight7 = np.array([])
all_BIC7 = np.array([]) 
all_negLL7 = np.array([]) 
all_trials7 = np.array([]) 

av_eps7 = np.array([]) 
av_alpha7 = np.array([]) 
av_weight7 = np.array([]) 
av_BIC7 = np.array([]) 
av_negLL7 = np.array([]) 
av_trials7 = np.array([])
results7 = {}

all_eps8 = np.array([]) 
all_alpha8 = np.array([]) 
all_BIC8= np.array([]) 
all_negLL8 = np.array([]) 
all_trials8 = np.array([]) 

av_eps8 = np.array([]) 
av_alpha8 = np.array([]) 
av_BIC8 = np.array([]) 
av_negLL8 = np.array([]) 
av_trials8 = np.array([])
results8 = {}


all_eps9 = np.array([]) 
all_alpha9 = np.array([]) 
all_unchosen9 = np.array([]) 
all_weight9 = np.array([])
all_BIC9= np.array([]) 
all_negLL9 = np.array([]) 
all_trials9 = np.array([]) 

av_eps9 = np.array([]) 
av_alpha9 = np.array([]) 
av_unchosen9 = np.array([]) 
av_weight9 = np.array([])
av_BIC9 = np.array([]) 
av_negLL9 = np.array([]) 
av_trials9 = np.array([])
results9 = {}



all_eps10 = np.array([]) 
all_alpha10 = np.array([]) 
all_unchosen10 = np.array([]) 
all_weight10 = np.array([])
all_BIC10= np.array([]) 
all_negLL10 = np.array([]) 
all_trials10 = np.array([]) 

av_eps10 = np.array([]) 
av_alpha10 = np.array([]) 
av_unchosen10 = np.array([]) 
av_weight10 = np.array([])
av_BIC10 = np.array([]) 
av_negLL10 = np.array([]) 
av_trials10 = np.array([])
results10 = {}



all_eps11 = np.array([]) 
all_alpha11 = np.array([]) 
all_weight11 = np.array([])
all_BIC11 = np.array([]) 
all_negLL11 = np.array([]) 
all_trials11 = np.array([]) 

av_eps11 = np.array([]) 
av_alpha11 = np.array([]) 
av_weight11 = np.array([])
av_BIC11 = np.array([]) 
av_negLL11 = np.array([]) 
av_trials11 = np.array([])
results11 = {}



all_eps12 = np.array([]) 
all_alpha12 = np.array([]) 
all_weight12 = np.array([])
all_BIC12 = np.array([]) 
all_negLL12 = np.array([]) 
all_trials12 = np.array([]) 

av_eps12 = np.array([]) 
av_alpha12 = np.array([]) 
av_weight12 = np.array([]) 
av_BIC12 = np.array([]) 
av_negLL12 = np.array([]) 
av_trials12 = np.array([])
results12 = {}


for count, folder in enumerate(maps):
    print('starting rat', count)
    rat = 0
    count=count+1
    subj_dir = os.path.join(file_dir,f'{folder}')
    subj_data_list = os.listdir(subj_dir)
    
   
    sub_eps1 = np.array([]) 
    sub_alpha1 = np.array([]) 
    sub_unchosen1 = np.array([]) 
    sub_BIC1 = np.array([]) 
    sub_negLL1 = np.array([]) 
    sub_trials1 = np.array([])


    sub_eps2 = np.array([]) 
    sub_alpha2 = np.array([]) 
    sub_unchosen2 = np.array([])
    sub_weight2 = np.array([])
    sub_BIC2 = np.array([]) 
    sub_negLL2 = np.array([]) 
    sub_trials2 = np.array([])


    sub_eps3 = np.array([]) 
    sub_alpha3 = np.array([]) 
    sub_unchosen3 = np.array([]) 
    sub_BIC3 = np.array([]) 
    sub_negLL3 = np.array([]) 
    sub_trials3 = np.array([])


    sub_eps4 = np.array([]) 
    sub_alpha4 = np.array([]) 
    sub_unchosen4 = np.array([])
    sub_weight4 = np.array([])
    sub_BIC4 = np.array([]) 
    sub_negLL4 = np.array([]) 
    sub_trials4 = np.array([])

    sub_eps5 = np.array([]) 
    sub_alpha5 = np.array([]) 
    sub_weight5 = np.array([])
    sub_BIC5 = np.array([]) 
    sub_negLL5 = np.array([]) 
    sub_trials5 = np.array([])

    sub_eps6 = np.array([]) 
    sub_alpha6 = np.array([]) 
    sub_BIC6 = np.array([]) 
    sub_negLL6 = np.array([]) 
    sub_trials6 = np.array([])

    sub_eps7 = np.array([]) 
    sub_alpha7 = np.array([]) 
    sub_weight7 = np.array([])
    sub_BIC7 = np.array([]) 
    sub_negLL7 = np.array([]) 
    sub_trials7 = np.array([])

    sub_eps8 = np.array([]) 
    sub_alpha8= np.array([]) 
    sub_BIC8 = np.array([]) 
    sub_negLL8 = np.array([]) 
    sub_trials8 = np.array([])

    
    sub_eps9 = np.array([]) 
    sub_alpha9 = np.array([]) 
    sub_unchosen9 = np.array([])
    sub_weight9 = np.array([])
    sub_BIC9 = np.array([]) 
    sub_negLL9 = np.array([]) 
    sub_trials9 = np.array([])
    

    sub_eps10 = np.array([]) 
    sub_alpha10 = np.array([]) 
    sub_unchosen10 = np.array([])
    sub_weight10 = np.array([])
    sub_BIC10 = np.array([]) 
    sub_negLL10 = np.array([]) 
    sub_trials10 = np.array([])


    sub_eps11 = np.array([]) 
    sub_alpha11 = np.array([]) 
    sub_weight11 = np.array([])
    sub_BIC11 = np.array([]) 
    sub_negLL11 = np.array([]) 
    sub_trials11 = np.array([])


    sub_eps12 = np.array([]) 
    sub_alpha12 = np.array([]) 
    sub_weight12 = np.array([])
    sub_BIC12 = np.array([]) 
    sub_negLL12 = np.array([]) 
    sub_trials12 = np.array([])




    for file_name in subj_data_list:
        print('rat', count, 'file', file_name)

        rat_nr = np.append(rat_nr, count)
        file_nr = np.append(file_nr, file_name)

        data_dir = os.path.join(subj_dir, file_name)
        Data = pd.read_excel(data_dir)
        data = Data.to_numpy()
        
        trials = np.shape(data)[0]

        k= data[:,0]
        k= np.add(k,-1)
        k= k.astype(int)
                
        r= data[:,2]
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

        seedings = int(np.sum(k[5:16])*np.sum(r[16:52])*(count+1))
        print('random seed for', count, 'is', seedings)

        random.seed(seedings)
        np.random.seed(seedings)

        
        #fit model 1: RW, fitting eps, alpha and lambda unchosen
        result1 = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
        negLL1 = result1.fun
        param_fits1 = result1.x
        BIC1 = len(bounds1) * np.log(trials) + 2*negLL1


        #this one saves al parameters of all subjects and all days
        all_eps1 = np.append(all_eps1, param_fits1[1])
        all_alpha1 = np.append(all_alpha1, param_fits1[0])
        all_unchosen1 = np.append(all_unchosen1, param_fits1[2])
        all_BIC1 = np.append(all_BIC1, BIC1)
        all_negLL1 = np.append(all_negLL1, negLL1)
        all_trials1 = np.append(all_trials1, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps1 = np.append(sub_eps1, param_fits1[1])
        sub_alpha1 = np.append(sub_alpha1, param_fits1[0])
        sub_unchosen1 = np.append(sub_unchosen1, param_fits1[2])
        sub_BIC1 = np.append(sub_BIC1, BIC1)
        sub_negLL1 = np.append(sub_negLL1, negLL1)
        sub_trials1 = np.append(sub_trials1, trials)

        print('model 1 done')





        #fit model 2
        result2 = differential_evolution(negll_2RW_eps_alpha_uc, bounds=bounds2, args=(k,r), strategy=strategy)
        negLL2 = result2.fun
        param_fits2 = result2.x
        BIC2 = len(bounds2) * np.log(trials) + 2*negLL2


        #this one saves al parameters of all subjects and all days
        all_eps2 = np.append(all_eps2, param_fits2[1])
        all_alpha2 = np.append(all_alpha2, param_fits2[0])
        all_unchosen2 = np.append(all_unchosen2, param_fits2[2])
        all_weight2 = np.append(all_weight2, param_fits2[3])
        all_BIC2 = np.append(all_BIC2, BIC2)
        all_negLL2 = np.append(all_negLL2, negLL2)
        all_trials2 = np.append(all_trials2, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps2 = np.append(sub_eps2, param_fits2[1])
        sub_alpha2 = np.append(sub_alpha2, param_fits2[0])
        sub_unchosen2 = np.append(sub_unchosen2, param_fits2[2])
        sub_weight2 = np.append(sub_weight2, param_fits2[3])
        sub_BIC2 = np.append(sub_BIC2, BIC2)
        sub_negLL2 = np.append(sub_negLL2, negLL2)
        sub_trials2 = np.append(sub_trials2, trials)


        print('model 2 done')



        #fit model 3
        result3 = differential_evolution(negll_3RW_eps_alpha_uc, bounds=bounds3, args=(k,r), strategy=strategy)
        negLL3 = result3.fun
        param_fits3 = result3.x
        BIC3 = len(bounds3) * np.log(trials) + 2*negLL3


        #this one saves al parameters of all subjects and all days
        all_eps3 = np.append(all_eps3, param_fits3[1])
        all_alpha3 = np.append(all_alpha3, param_fits3[0])
        all_unchosen3 = np.append(all_unchosen3, param_fits3[2])
        all_BIC3 = np.append(all_BIC3, BIC3)
        all_negLL3 = np.append(all_negLL3, negLL3)
        all_trials3 = np.append(all_trials3, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps3 = np.append(sub_eps3, param_fits3[1])
        sub_alpha3 = np.append(sub_alpha3, param_fits3[0])
        sub_unchosen3 = np.append(sub_unchosen3, param_fits3[2])
        sub_BIC3 = np.append(sub_BIC3, BIC3)
        sub_negLL3 = np.append(sub_negLL3, negLL3)
        sub_trials3 = np.append(sub_trials3, trials)


        print('model 3 done')




        #fit model 4
        result4 = differential_evolution(negll_4RW_eps_alpha_uc, bounds=bounds4, args=(k,r), strategy=strategy)
        negLL4 = result4.fun
        param_fits4 = result4.x
        BIC4 = len(bounds4) * np.log(trials) + 2*negLL4


        #this one saves al parameters of all subjects and all days
        all_eps4 = np.append(all_eps4, param_fits4[1])
        all_alpha4 = np.append(all_alpha4, param_fits4[0])
        all_unchosen4 = np.append(all_unchosen4, param_fits4[2])
        all_weight4 = np.append(all_weight4, param_fits4[2])
        all_BIC4 = np.append(all_BIC4, BIC4)
        all_negLL4 = np.append(all_negLL4, negLL4)
        all_trials4 = np.append(all_trials4, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps4 = np.append(sub_eps4, param_fits4[1])
        sub_alpha4 = np.append(sub_alpha4, param_fits4[0])
        sub_unchosen4 = np.append(sub_unchosen4, param_fits4[2])
        sub_weight4 = np.append(sub_weight4, param_fits4[3])
        sub_BIC4 = np.append(sub_BIC4, BIC4)
        sub_negLL4 = np.append(sub_negLL4, negLL4)
        sub_trials4 = np.append(sub_trials4, trials)


        print('model 4 done')


        #fit model 5
        result5 = differential_evolution(negll_5RW_eps_alpha_uc, bounds=bounds5, args=(k,r), strategy=strategy)
        negLL5 = result5.fun
        param_fits5 = result5.x
        BIC5 = len(bounds5) * np.log(trials) + 2*negLL5


        #this one saves al parameters of all subjects and all days
        all_eps5 = np.append(all_eps5, param_fits5[1])
        all_alpha5 = np.append(all_alpha5, param_fits5[0])
        all_weight5 = np.append(all_weight5, param_fits5[2])
        all_BIC5 = np.append(all_BIC5, BIC5)
        all_negLL5 = np.append(all_negLL5, negLL5)
        all_trials5 = np.append(all_trials5, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps5 = np.append(sub_eps5, param_fits5[1])
        sub_alpha5 = np.append(sub_alpha5, param_fits5[0])
        sub_weight5 = np.append(sub_weight5, param_fits5[2])
        sub_BIC5 = np.append(sub_BIC5, BIC5)
        sub_negLL5 = np.append(sub_negLL5, negLL5)
        sub_trials5 = np.append(sub_trials5, trials)


        print('model 5 done')



        #fit model 6
        result6 = differential_evolution(negll_6RW_eps_alpha_uc, bounds=bounds6, args=(k,r), strategy=strategy)
        negLL6 = result6.fun
        param_fits6 = result6.x
        BIC6 = len(bounds6) * np.log(trials) + 2*negLL6


        #this one saves al parameters of all subjects and all days
        all_eps6 = np.append(all_eps6, param_fits6[1])
        all_alpha6 = np.append(all_alpha6, param_fits6[0])
        all_BIC6 = np.append(all_BIC6, BIC6)
        all_negLL6 = np.append(all_negLL6, negLL6)
        all_trials6 = np.append(all_trials6, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps6 = np.append(sub_eps6, param_fits6[1])
        sub_alpha6 = np.append(sub_alpha6, param_fits6[0])
        sub_BIC6 = np.append(sub_BIC6, BIC6)
        sub_negLL6 = np.append(sub_negLL6, negLL6)
        sub_trials6 = np.append(sub_trials6, trials)


        print('model 6 done')



        #fit model 7
        result7 = differential_evolution(negll_7RW_eps_alpha_uc, bounds=bounds7, args=(k,r), strategy=strategy)
        negLL7 = result7.fun
        param_fits7 = result7.x
        BIC7 = len(bounds7) * np.log(trials) + 2*negLL7


        #this one saves al parameters of all subjects and all days
        all_eps7 = np.append(all_eps7, param_fits7[1])
        all_alpha7 = np.append(all_alpha7, param_fits7[0])
        all_weight7 = np.append(all_weight7, param_fits7[2])
        all_BIC7 = np.append(all_BIC7, BIC7)
        all_negLL7 = np.append(all_negLL7, negLL7)
        all_trials7 = np.append(all_trials7, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps7 = np.append(sub_eps7, param_fits7[1])
        sub_alpha7 = np.append(sub_alpha7, param_fits7[0])
        sub_weight7 = np.append(sub_weight7, param_fits7[2])
        sub_BIC7 = np.append(sub_BIC7, BIC7)
        sub_negLL7 = np.append(sub_negLL7, negLL7)
        sub_trials7 = np.append(sub_trials7, trials)


        print('model 7 done')



        #fit model 8
        result8 = differential_evolution(negll_8RW_eps_alpha_uc, bounds=bounds8, args=(k,r), strategy=strategy)
        negLL8 = result8.fun
        param_fits8 = result8.x
        BIC8 = len(bounds8) * np.log(trials) + 2*negLL8


        #this one saves al parameters of all subjects and all days
        all_eps8 = np.append(all_eps8, param_fits8[1])
        all_alpha8 = np.append(all_alpha8, param_fits8[0])
        all_BIC8 = np.append(all_BIC8, BIC8)
        all_negLL8 = np.append(all_negLL8, negLL8)
        all_trials8 = np.append(all_trials8, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps8 = np.append(sub_eps8, param_fits8[1])
        sub_alpha8 = np.append(sub_alpha8, param_fits8[0])
        sub_BIC8 = np.append(sub_BIC8, BIC8)
        sub_negLL8 = np.append(sub_negLL8, negLL8)
        sub_trials8 = np.append(sub_trials8, trials)


        print('model 8 done')


        

        #fit model 9
        result9 = differential_evolution(negll_9RW_eps_alpha_uc, bounds=bounds9, args=(k,r), strategy=strategy)
        negLL9 = result9.fun
        param_fits9 = result9.x
        BIC9 = len(bounds9) * np.log(trials) + 2*negLL9


        #this one saves al parameters of all subjects and all days
        all_eps9 = np.append(all_eps9, param_fits9[1])
        all_alpha9 = np.append(all_alpha9, param_fits9[0])
        all_unchosen9 = np.append(all_unchosen9, param_fits9[2])
        all_weight9 = np.append(all_weight9, param_fits9[3])
        all_BIC9 = np.append(all_BIC9, BIC9)
        all_negLL9 = np.append(all_negLL9, negLL9)
        all_trials9 = np.append(all_trials9, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps9 = np.append(sub_eps9, param_fits9[1])
        sub_alpha9 = np.append(sub_alpha9, param_fits9[0])
        sub_unchosen9 = np.append(sub_unchosen9, param_fits9[2])
        sub_weight9 = np.append(sub_weight9, param_fits9[3])
        sub_BIC9 = np.append(sub_BIC9, BIC9)
        sub_negLL9 = np.append(sub_negLL9, negLL9)
        sub_trials9 = np.append(sub_trials9, trials)


        print('model 9 done')



        #fit model 10
        result10 = differential_evolution(negll_10RW_eps_alpha_uc, bounds=bounds10, args=(k,r), strategy=strategy)
        negLL10 = result10.fun
        param_fits10 = result10.x
        BIC10 = len(bounds10) * np.log(trials) + 2*negLL10


        #this one saves al parameters of all subjects and all days
        all_eps10 = np.append(all_eps10, param_fits10[1])
        all_alpha10 = np.append(all_alpha10, param_fits10[0])
        all_unchosen10 = np.append(all_unchosen10, param_fits10[2])
        all_weight10 = np.append(all_weight10, param_fits10[3])
        all_BIC10 = np.append(all_BIC10, BIC10)
        all_negLL10 = np.append(all_negLL10, negLL10)
        all_trials10 = np.append(all_trials10, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps10 = np.append(sub_eps10, param_fits10[1])
        sub_alpha10 = np.append(sub_alpha10, param_fits10[0])
        sub_unchosen10 = np.append(sub_unchosen10, param_fits10[2])
        sub_weight10 = np.append(sub_weight10, param_fits10[3])
        sub_BIC10 = np.append(sub_BIC10, BIC10)
        sub_negLL10 = np.append(sub_negLL10, negLL10)
        sub_trials10 = np.append(sub_trials10, trials)


        print('model 10 done')



        #fit model 11
        result11 = differential_evolution(negll_11RW_eps_alpha_uc, bounds=bounds11, args=(k,r), strategy=strategy)
        negLL11 = result11.fun
        param_fits11 = result11.x
        BIC11 = len(bounds11) * np.log(trials) + 2*negLL11


        #this one saves al parameters of all subjects and all days
        all_eps11 = np.append(all_eps11, param_fits11[1])
        all_alpha11 = np.append(all_alpha11, param_fits11[0])
        all_weight11 = np.append(all_weight11, param_fits11[2])
        all_BIC11 = np.append(all_BIC11, BIC11)
        all_negLL11 = np.append(all_negLL11, negLL11)
        all_trials11 = np.append(all_trials11, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps11 = np.append(sub_eps11, param_fits11[1])
        sub_alpha11 = np.append(sub_alpha11, param_fits11[0])
        sub_weight11 = np.append(sub_weight11, param_fits11[2])
        sub_BIC11 = np.append(sub_BIC11, BIC11)
        sub_negLL11 = np.append(sub_negLL11, negLL11)
        sub_trials11 = np.append(sub_trials11, trials)


        print('model 11 done')


        #fit model 12
        result12 = differential_evolution(negll_12RW_eps_alpha_uc, bounds=bounds12, args=(k,r), strategy=strategy)
        negLL12 = result12.fun
        param_fits12 = result12.x
        BIC12 = len(bounds12) * np.log(trials) + 2*negLL12


        #this one saves al parameters of all subjects and all days
        all_eps12 = np.append(all_eps12, param_fits12[1])
        all_alpha12 = np.append(all_alpha12, param_fits12[0])
        all_weight12 = np.append(all_weight12, param_fits12[2])
        all_BIC12 = np.append(all_BIC12, BIC12)
        all_negLL12 = np.append(all_negLL12, negLL12)
        all_trials12 = np.append(all_trials12, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps12 = np.append(sub_eps12, param_fits12[1])
        sub_alpha12 = np.append(sub_alpha12, param_fits12[0])
        sub_weight12 = np.append(sub_weight12, param_fits12[2])
        sub_BIC12 = np.append(sub_BIC12, BIC12)
        sub_negLL12 = np.append(sub_negLL12, negLL12)
        sub_trials12 = np.append(sub_trials12, trials)


        print('model 12 done')



   
    sub_av_eps1 = np.mean(sub_eps1)
    sub_av_alpha1 = np.mean(sub_alpha1)
    sub_av_unchosen1 = np.mean(sub_unchosen1)
    sub_av_BIC1 = np.mean(sub_BIC1)
    sub_av_negLL1 = np.mean(sub_negLL1)
    sub_av_trials1 = np.mean(sub_trials1)

    sub_av_eps2 = np.mean(sub_eps2)
    sub_av_alpha2 = np.mean(sub_alpha2)
    sub_av_unchosen2 = np.mean(sub_unchosen2)
    sub_av_weight2 = np.mean(sub_weight2)   
    sub_av_BIC2 = np.mean(sub_BIC2)
    sub_av_negLL2 = np.mean(sub_negLL2)
    sub_av_trials2 = np.mean(sub_trials2)


    sub_av_eps3 = np.mean(sub_eps3)
    sub_av_alpha3 = np.mean(sub_alpha3)
    sub_av_unchosen3 = np.mean(sub_unchosen3)
    sub_av_BIC3 = np.mean(sub_BIC3)
    sub_av_negLL3 = np.mean(sub_negLL3)
    sub_av_trials3 = np.mean(sub_trials3)

    sub_av_eps4 = np.mean(sub_eps4)
    sub_av_alpha4= np.mean(sub_alpha4)
    sub_av_unchosen4 = np.mean(sub_unchosen4)
    sub_av_weight4 = np.mean(sub_weight4)   
    sub_av_BIC4 = np.mean(sub_BIC4)
    sub_av_negLL4 = np.mean(sub_negLL4)
    sub_av_trials4 = np.mean(sub_trials4)


    sub_av_eps5 = np.mean(sub_eps5)
    sub_av_alpha5 = np.mean(sub_alpha5)
    sub_av_weight5 = np.mean(sub_weight5)   
    sub_av_BIC5 = np.mean(sub_BIC5)
    sub_av_negLL5 = np.mean(sub_negLL5)
    sub_av_trials5 = np.mean(sub_trials5)

    sub_av_eps6 = np.mean(sub_eps6)
    sub_av_alpha6 = np.mean(sub_alpha6)
    sub_av_BIC6 = np.mean(sub_BIC6)
    sub_av_negLL6 = np.mean(sub_negLL6)
    sub_av_trials6 = np.mean(sub_trials6)


    sub_av_eps7 = np.mean(sub_eps7)
    sub_av_alpha7 = np.mean(sub_alpha7)
    sub_av_weight7 = np.mean(sub_weight7)   
    sub_av_BIC7 = np.mean(sub_BIC7)
    sub_av_negLL7 = np.mean(sub_negLL7)
    sub_av_trials7 = np.mean(sub_trials7)

    sub_av_eps8 = np.mean(sub_eps8)
    sub_av_alpha8 = np.mean(sub_alpha8)
    sub_av_BIC8 = np.mean(sub_BIC8)
    sub_av_negLL8 = np.mean(sub_negLL8)
    sub_av_trials8 = np.mean(sub_trials8)


    sub_av_eps9 = np.mean(sub_eps9)
    sub_av_alpha9 = np.mean(sub_alpha9)
    sub_av_unchosen9 = np.mean(sub_unchosen9)
    sub_av_weight9 = np.mean(sub_weight9)   
    sub_av_BIC9 = np.mean(sub_BIC9)
    sub_av_negLL9 = np.mean(sub_negLL9)
    sub_av_trials9 = np.mean(sub_trials9)


    sub_av_eps10 = np.mean(sub_eps10)
    sub_av_alpha10 = np.mean(sub_alpha10)
    sub_av_unchosen10 = np.mean(sub_unchosen10)
    sub_av_weight10 = np.mean(sub_weight10)   
    sub_av_BIC10 = np.mean(sub_BIC10)
    sub_av_negLL10 = np.mean(sub_negLL10)
    sub_av_trials10 = np.mean(sub_trials10)


    sub_av_eps11 = np.mean(sub_eps11)
    sub_av_alpha11 = np.mean(sub_alpha11)
    sub_av_weight11 = np.mean(sub_weight11)   
    sub_av_BIC11 = np.mean(sub_BIC11)
    sub_av_negLL11 = np.mean(sub_negLL11)
    sub_av_trials11 = np.mean(sub_trials11)

    sub_av_eps12 = np.mean(sub_eps12)
    sub_av_alpha12 = np.mean(sub_alpha12)
    sub_av_weight12 = np.mean(sub_weight12)   
    sub_av_BIC12 = np.mean(sub_BIC12)
    sub_av_negLL12 = np.mean(sub_negLL12)
    sub_av_trials12 = np.mean(sub_trials12)




   
    av_eps1 = np.append(av_eps1, sub_av_eps1)
    av_alpha1 = np.append(av_alpha1, sub_av_alpha1)
    av_unchosen1 = np.append(av_unchosen1, sub_av_unchosen1)
    av_BIC1 = np.append(av_BIC1, sub_av_BIC1)
    av_negLL1 = np.append(av_negLL1, sub_av_negLL1)
    av_trials1= np.append(av_trials1, sub_av_trials1)

    av_eps2 = np.append(av_eps2, sub_av_eps2)
    av_alpha2 = np.append(av_alpha2, sub_av_alpha2)
    av_unchosen2 = np.append(av_unchosen2, sub_av_unchosen2)
    av_weight2 = np.append(av_weight2, sub_av_weight2)
    av_BIC2 = np.append(av_BIC2, sub_av_BIC2)
    av_negLL2 = np.append(av_negLL2, sub_av_negLL2)
    av_trials2 = np.append(av_trials2, sub_av_trials2)


    av_eps3 = np.append(av_eps3, sub_av_eps3)
    av_alpha3 = np.append(av_alpha3, sub_av_alpha3)
    av_unchosen3 = np.append(av_unchosen3, sub_av_unchosen3)
    av_BIC3 = np.append(av_BIC3, sub_av_BIC3)
    av_negLL3 = np.append(av_negLL3, sub_av_negLL3)
    av_trials3= np.append(av_trials3, sub_av_trials3)

    av_eps4 = np.append(av_eps4, sub_av_eps4)
    av_alpha4 = np.append(av_alpha4, sub_av_alpha4)
    av_unchosen4 = np.append(av_unchosen4, sub_av_unchosen4)
    av_weight4 = np.append(av_weight4, sub_av_weight4)
    av_BIC4 = np.append(av_BIC4, sub_av_BIC4)
    av_negLL4 = np.append(av_negLL4, sub_av_negLL4)
    av_trials4 = np.append(av_trials4, sub_av_trials4)

    av_eps5 = np.append(av_eps5, sub_av_eps5)
    av_alpha5 = np.append(av_alpha5, sub_av_alpha5)
    av_weight5 = np.append(av_weight5, sub_av_weight5)
    av_BIC5 = np.append(av_BIC5, sub_av_BIC5)
    av_negLL5 = np.append(av_negLL5, sub_av_negLL5)
    av_trials5 = np.append(av_trials5, sub_av_trials5)

    av_eps6 = np.append(av_eps6, sub_av_eps6)
    av_alpha6 = np.append(av_alpha6, sub_av_alpha6)
    av_BIC6 = np.append(av_BIC6, sub_av_BIC6)
    av_negLL6 = np.append(av_negLL6, sub_av_negLL6)
    av_trials6 = np.append(av_trials6, sub_av_trials6)


    av_eps7 = np.append(av_eps7, sub_av_eps7)
    av_alpha7 = np.append(av_alpha7, sub_av_alpha7)
    av_weight7 = np.append(av_weight7, sub_av_weight7)
    av_BIC7 = np.append(av_BIC7, sub_av_BIC7)
    av_negLL7 = np.append(av_negLL7, sub_av_negLL7)
    av_trials7 = np.append(av_trials7, sub_av_trials7)

    av_eps8 = np.append(av_eps8, sub_av_eps8)
    av_alpha8 = np.append(av_alpha8, sub_av_alpha8)
    av_BIC8 = np.append(av_BIC8, sub_av_BIC8)
    av_negLL8 = np.append(av_negLL8, sub_av_negLL8)
    av_trials8 = np.append(av_trials8, sub_av_trials8)
  
    av_eps9 = np.append(av_eps9, sub_av_eps9)
    av_alpha9 = np.append(av_alpha9, sub_av_alpha9)
    av_unchosen9 = np.append(av_unchosen9, sub_av_unchosen9)
    av_weight9 = np.append(av_weight9, sub_av_weight9)
    av_BIC9 = np.append(av_BIC9, sub_av_BIC9)
    av_negLL9 = np.append(av_negLL9, sub_av_negLL9)
    av_trials9 = np.append(av_trials9, sub_av_trials9)


    av_eps10 = np.append(av_eps10, sub_av_eps10)
    av_alpha10 = np.append(av_alpha10, sub_av_alpha10)
    av_unchosen10 = np.append(av_unchosen10, sub_av_unchosen10)
    av_weight10 = np.append(av_weight10, sub_av_weight10)
    av_BIC10 = np.append(av_BIC10, sub_av_BIC10)
    av_negLL10 = np.append(av_negLL10, sub_av_negLL10)
    av_trials10 = np.append(av_trials10, sub_av_trials10)

    av_eps11 = np.append(av_eps11, sub_av_eps11)
    av_alpha11 = np.append(av_alpha11, sub_av_alpha11)
    av_weight11 = np.append(av_weight11, sub_av_weight11)
    av_BIC11 = np.append(av_BIC11, sub_av_BIC11)
    av_negLL11 = np.append(av_negLL11, sub_av_negLL11)
    av_trials11 = np.append(av_trials11, sub_av_trials11)


    av_eps12 = np.append(av_eps12, sub_av_eps12)
    av_alpha12 = np.append(av_alpha12, sub_av_alpha12)
    av_weight12 = np.append(av_weight12, sub_av_weight12)
    av_BIC12 = np.append(av_BIC12, sub_av_BIC12)
    av_negLL12 = np.append(av_negLL12, sub_av_negLL12)
    av_trials12 = np.append(av_trials12, sub_av_trials12)


save_dir = os.path.join(data_dir1, 'Rat_output_versions_lambda')


all_results1 = {}
all_results2 = {}
all_results3 = {}
all_results4 = {}
all_results5 = {}
all_results6 = {}
all_results7 = {}
all_results8 = {}

all_results9 = {}
all_results10 = {}
all_results11 = {}
all_results12 = {}

all_results1[f'eps'] = all_eps1
all_results1[f'alpha'] = all_alpha1
all_results1[f'unchosen'] = all_unchosen1
all_results1[f'BIC'] = all_BIC1
all_results1[f'negLL'] = all_negLL1
all_results1[f'trials'] = all_trials1  
all_results1['rat'] = rat_nr
all_results1['file'] = file_nr

results1[f'eps'] = av_eps1
results1[f'alpha'] = av_alpha1
results1[f'unchosen'] = av_unchosen1
results1[f'BIC'] = av_BIC1
results1[f'negLL'] = av_negLL1
results1[f'trials'] = av_trials1


dict_name = f'1Rats_av_eps-alpha-uc.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results1)
df.to_excel(dict_save)

dict_name = f'1Rats_all_eps-alpha-uc.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results1)
df.to_excel(dict_save)




results2[f'eps'] = av_eps2
results2[f'alpha'] = av_alpha2
results2[f'unchosen'] = av_unchosen2
results2[f'weight'] = av_weight2
results2[f'BIC'] = av_BIC2
results2[f'negLL'] = av_negLL2
results2[f'trials'] = av_trials2


all_results2[f'eps'] = all_eps2
all_results2[f'alpha'] = all_alpha2
all_results2[f'unchosen'] = all_unchosen2
all_results2[f'weight'] = all_weight2
all_results2[f'BIC'] = all_BIC2
all_results2[f'negLL'] = all_negLL2
all_results2[f'trials'] = all_trials2  
all_results2['rat'] = rat_nr
all_results2['file'] = file_nr

dict_name = f'2Rats_av_seperate_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results2)
df.to_excel(dict_save)

dict_name = f'2Rats_all_seperate_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results2)
df.to_excel(dict_save)




all_results3[f'eps'] = all_eps3
all_results3[f'alpha'] = all_alpha3
all_results3[f'unchosen'] = all_unchosen3
all_results3[f'BIC'] = all_BIC3
all_results3[f'negLL'] = all_negLL3
all_results3[f'trials'] = all_trials3 
all_results3['rat'] = rat_nr
all_results3['file'] = file_nr

results3[f'eps'] = av_eps3
results3[f'alpha'] = av_alpha3
results3[f'unchosen'] = av_unchosen3
results3[f'BIC'] = av_BIC3
results3[f'negLL'] = av_negLL3
results3[f'trials'] = av_trials3


dict_name = f'3Rats_av_seperate_uc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results3)
df.to_excel(dict_save)

dict_name = f'3Rats_all_seperate_uc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results3)
df.to_excel(dict_save)




results4[f'eps'] = av_eps4
results4[f'alpha'] = av_alpha4
results4[f'unchosen'] = av_unchosen4
results4[f'weight'] = av_weight4
results4[f'BIC'] = av_BIC4
results4[f'negLL'] = av_negLL4
results4[f'trials'] = av_trials4


all_results4[f'eps'] = all_eps4
all_results4[f'alpha'] = all_alpha4
all_results4[f'unchosen'] = all_unchosen4
all_results4[f'weight'] = all_weight4
all_results4[f'BIC'] = all_BIC4
all_results4[f'negLL'] = all_negLL4
all_results4[f'trials'] = all_trials4 
all_results4['rat'] = rat_nr
all_results4['file'] = file_nr

dict_name = f'4Rats_av_RWuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results4)
df.to_excel(dict_save)

dict_name = f'4Rats_all_RWuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results4)
df.to_excel(dict_save)



results5[f'eps'] = av_eps5
results5[f'alpha'] = av_alpha5
results5[f'weight'] = av_weight5
results5[f'BIC'] = av_BIC5
results5[f'negLL'] = av_negLL5
results5[f'trials'] = av_trials5


all_results5[f'eps'] = all_eps5
all_results5[f'alpha'] = all_alpha5
all_results5[f'weight'] = all_weight5
all_results5[f'BIC'] = all_BIC5
all_results5[f'negLL'] = all_negLL5
all_results5[f'trials'] = all_trials5
all_results5['rat'] = rat_nr
all_results5['file'] = file_nr

dict_name = f'5Rats_av_posuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results5)
df.to_excel(dict_save)

dict_name = f'5Rats_all_posuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results5)
df.to_excel(dict_save)



results6[f'eps'] = av_eps6
results6[f'alpha'] = av_alpha6
results6[f'BIC'] = av_BIC6
results6[f'negLL'] = av_negLL6
results6[f'trials'] = av_trials6


all_results6[f'eps'] = all_eps6
all_results6[f'alpha'] = all_alpha6
all_results6[f'BIC'] = all_BIC6
all_results6[f'negLL'] = all_negLL6
all_results6[f'trials'] = all_trials6
all_results6['rat'] = rat_nr
all_results6['file'] = file_nr

dict_name = f'6Rats_av_posuc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results6)
df.to_excel(dict_save)

dict_name = f'6Rats_all_posuc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results6)
df.to_excel(dict_save)






results7[f'eps'] = av_eps7
results7[f'alpha'] = av_alpha7
results7[f'weight'] = av_weight7
results7[f'BIC'] = av_BIC7
results7[f'negLL'] = av_negLL7
results7[f'trials'] = av_trials7


all_results7[f'eps'] = all_eps7
all_results7[f'alpha'] = all_alpha7
all_results7[f'weight'] = all_weight7
all_results7[f'BIC'] = all_BIC7
all_results7[f'negLL'] = all_negLL7
all_results7[f'trials'] = all_trials7
all_results7['rat'] = rat_nr
all_results7['file'] = file_nr

dict_name = f'7Rats_av_neguc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results7)
df.to_excel(dict_save)

dict_name = f'7Rats_all_neguc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results7)
df.to_excel(dict_save)



results8[f'eps'] = av_eps8
results8[f'alpha'] = av_alpha8
results8[f'BIC'] = av_BIC8
results8[f'negLL'] = av_negLL8
results8[f'trials'] = av_trials8


all_results8[f'eps'] = all_eps8
all_results8[f'alpha'] = all_alpha8
all_results8[f'BIC'] = all_BIC8
all_results8[f'negLL'] = all_negLL8
all_results8[f'trials'] = all_trials8
all_results8['rat'] = rat_nr
all_results8['file'] = file_nr

dict_name = f'8Rats_av_neguc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results8)
df.to_excel(dict_save)

dict_name = f'8Rats_all_neguc_noweight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results8)
df.to_excel(dict_save)






results9[f'eps'] = av_eps9
results9[f'alpha'] = av_alpha9
results9[f'unchosen'] = av_unchosen9
results9[f'weight'] = av_weight9
results9[f'BIC'] = av_BIC9
results9[f'negLL'] = av_negLL9
results9[f'trials'] = av_trials9


all_results9[f'eps'] = all_eps9
all_results9[f'alpha'] = all_alpha9
all_results9[f'unchosen'] = all_unchosen9
all_results9[f'weight'] = all_weight9
all_results9[f'BIC'] = all_BIC9
all_results9[f'negLL'] = all_negLL9
all_results9[f'trials'] = all_trials9
all_results9['rat'] = rat_nr
all_results9['file'] = file_nr

dict_name = f'9Rats_av_seperate_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results9)
df.to_excel(dict_save)

dict_name = f'9Rats_all_seperate_uc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results9)
df.to_excel(dict_save)





results10[f'eps'] = av_eps10
results10[f'alpha'] = av_alpha10
results10[f'unchosen'] = av_unchosen10
results10[f'weight'] = av_weight10
results10[f'BIC'] = av_BIC10
results10[f'negLL'] = av_negLL10
results10[f'trials'] = av_trials10


all_results10[f'eps'] = all_eps10
all_results10[f'alpha'] = all_alpha10
all_results10[f'unchosen'] = all_unchosen10
all_results10[f'weight'] = all_weight10
all_results10[f'BIC'] = all_BIC10
all_results10[f'negLL'] = all_negLL10
all_results10[f'trials'] = all_trials10
all_results10['rat'] = rat_nr
all_results10['file'] = file_nr

dict_name = f'10Rats_av_RWuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results10)
df.to_excel(dict_save)

dict_name = f'10Rats_all_RWuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results10)
df.to_excel(dict_save)



results11[f'eps'] = av_eps11
results11[f'alpha'] = av_alpha11
results11[f'weight'] = av_weight11
results11[f'BIC'] = av_BIC11
results11[f'negLL'] = av_negLL11
results11[f'trials'] = av_trials11


all_results11[f'eps'] = all_eps11
all_results11[f'alpha'] = all_alpha11
all_results11[f'weight'] = all_weight11
all_results11[f'BIC'] = all_BIC11
all_results11[f'negLL'] = all_negLL11
all_results11[f'trials'] = all_trials11
all_results11['rat'] = rat_nr
all_results11['file'] = file_nr

dict_name = f'11Rats_av_posuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results11)
df.to_excel(dict_save)

dict_name = f'11Rats_all_posuc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results11)
df.to_excel(dict_save)





results12[f'eps'] = av_eps12
results12[f'alpha'] = av_alpha12
results12[f'weight'] = av_weight12
results12[f'BIC'] = av_BIC12
results12[f'negLL'] = av_negLL12
results12[f'trials'] = av_trials12


all_results12[f'eps'] = all_eps12
all_results12[f'alpha'] = all_alpha12
all_results12[f'weight'] = all_weight12
all_results12[f'BIC'] = all_BIC12
all_results12[f'negLL'] = all_negLL12
all_results12[f'trials'] = all_trials12
all_results12['rat'] = rat_nr
all_results12['file'] = file_nr

dict_name = f'12Rats_av_neguc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results12)
df.to_excel(dict_save)

dict_name = f'12Rats_all_neguc_weight.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results12)
df.to_excel(dict_save)

BIC_all_data = {}
BIC_av_data = {}




BIC_all_data['eps-alpha-uc'] = all_BIC1
BIC_all_data['seperate-uc-weight'] = all_BIC2
BIC_all_data['seperate-uc-noweight'] = all_BIC3
BIC_all_data['RWuc-weight'] = all_BIC4
BIC_all_data['eps-alpha-posuc-weight'] = all_BIC5
BIC_all_data['eps-alpha-posuc'] = all_BIC6
BIC_all_data['eps-alpha-neguc-weight'] = all_BIC7
BIC_all_data['eps-alpha-neguc'] = all_BIC8

BIC_av_data['eps-alpha-uc'] = av_BIC1
BIC_av_data['seperate-uc-weight'] = av_BIC2
BIC_av_data['seperate-uc-noweight'] = av_BIC3
BIC_av_data['RWuc-weight'] = av_BIC4
BIC_av_data['eps-alpha-posuc-weight'] = av_BIC5
BIC_av_data['eps-alpha-posuc'] = av_BIC6
BIC_av_data['eps-alpha-neguc-weight'] = av_BIC7
BIC_av_data['eps-alpha-neguc'] = av_BIC8




BIC_all_data['seperate-uc-difweight'] = all_BIC9
BIC_all_data['RWuc-difweight'] = all_BIC10
BIC_all_data['eps-alpha-posuc-difweight'] = all_BIC11
BIC_all_data['eps-alpha-neguc-difweight'] = all_BIC12

BIC_av_data['seperate-uc-difweight'] = av_BIC9
BIC_av_data['RWuc-difweight'] = av_BIC10
BIC_av_data['eps-alpha-posuc-difweight'] = av_BIC11
BIC_av_data['eps-alpha-neguc-difweight'] = av_BIC12



dict_name = f'Rats_all_BIC.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=BIC_all_data)
df.to_excel(dict_save)

dict_name = f'Rats_av_BIC.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=BIC_av_data)
df.to_excel(dict_save)