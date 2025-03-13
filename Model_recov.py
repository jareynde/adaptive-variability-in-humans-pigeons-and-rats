#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Model recovery
Model one: eps, alpha, lambda
Model two: eps, alpha

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


################################################################################################################################
#First, simulate data with the models
################################################################################################################################
#simulation of Rescorla-Wagner model in an adversarial context according to rat experiment
def RW_rat_lambda_adversarial(Q_alpha, eps, unchosen, T, Q_int):
    #Q_alpha    --->        learning rate
    #eps        --->        epsilon
    #unchosen   --->        the added value to unchosen option values
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=5 #the amount of choice options
    K_seq = 125 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(5) for b in range(5) for c in range(5)])
    Freq1 = np.ones(K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        weighted_Freq1 = Freq1/sum(Freq1)
        if t < 2:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-2:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = weighted_Freq1[current_index]
            if current_freq < 0.0112:
                r[t] = 1
            else: r[t] = 0

            Freq1[current_index] = Freq1[current_index] + 1 
        if r[t] == 1:
            Freq1 = Freq1*0.95

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored

def RW_lambda_HP8(Q_alpha, eps, unchosen, T, Q_int):
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.ones(K_seq)*20

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < 21.6:
                r[t] = 1
            else: r[t] = 0
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
        if r[t] == 1:
            Freq1 = Freq1*0.984

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored

def RW_lambda_HP4(Q_alpha, eps, unchosen, T, Q_int):
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option

    K=4 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(4) for b in range(4) for c in range(4)])
    Freq1 = np.ones(K_seq)*20

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 2:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-2:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < 21.6:
                r[t] = 1
            else: r[t] = 0
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
        if r[t] == 1:
            Freq1 = Freq1*0.984

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored

def RW_lambda_HP2(Q_alpha, eps, unchosen, T, Q_int):
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option

    K=2 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a,b,c,d,e,f] for a in range(2) for b in range(2) for c in range(2) for d in range(2) for e in range(2) for f in range(2)])
    Freq1 = np.ones(K_seq)*20

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 5:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-5:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < 21.6:
                r[t] = 1
            else: r[t] = 0
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
        if r[t] == 1:
            Freq1 = Freq1*0.984

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored

################################################################################################################################
#Second, define the negative log likelihood function
################################################################################################################################
#model1
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

#model2
def negll_RW_eps_alpha(params, k, r):

    alpha, eps = params
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



    negLL = -np.sum(np.log(choice_prob)) 

    return negLL
    



################################################################################################################################
#Third, for each environment, we're gonna simulate some data from each model and then fit all model on all simulated data
################################################################################################################################
amount = 1000
T = 500
Q_int = 1
strategy = 'randtobest1bin'
bounds1=[(0,1), (0,1), (-3,3)]
bounds2=[(0,1), (0,1)]

######## RAT ADVERSARIAL CONTEXT ########
seeds = list(range(4*amount))  # Random seeds

BIC1 = np.zeros([amount,2])
BIC2 = np.zeros([amount,2])


### sim model 1: eps, alpha, unchosen ###
for i in range(amount):

    random.seed(seeds[i])
    np.random.seed(seeds[i])

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen = random.uniform(-3, 3)

    k, r, Q_k_stored = RW_rat_lambda_adversarial(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,1] = len(bounds2) * np.log(T) + 2*negLL




### sim model 2: eps, alpha ###

    random.seed(seeds[i+amount]+4857392)
    np.random.seed(seeds[i+amount]+231232)

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen=0

    k, r, Q_k_stored = RW_rat_lambda_adversarial(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,1] = len(bounds2) * np.log(T) + 2*negLL



    print(i, 'of', amount, 'done')



save_dir = os.path.dirname(os.path.abspath(__file__))
df_BIC1 = pd.DataFrame(BIC1)
df_BIC2 = pd.DataFrame(BIC2)

# Save the DataFrame to an Excel file
df_BIC1.to_excel(os.path.join(save_dir, '1rat_BIC_eps-alpha-lambda.xlsx'))
df_BIC2.to_excel(os.path.join(save_dir, '2rat_BIC_eps-alpha.xlsx'))











######## HP8 ADVERSARIAL CONTEXT ########
seeds = list(range(4*amount))  # Random seeds

BIC1 = np.zeros([amount,2])
BIC2 = np.zeros([amount,2])


### sim model 1: eps, alpha, unchosen ###
for i in range(amount):

    random.seed(seeds[i])
    np.random.seed(seeds[i])

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen = random.uniform(-3, 3)

    k, r, Q_k_stored = RW_lambda_HP8(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,1] = len(bounds2) * np.log(T) + 2*negLL




### sim model 2: eps, alpha ###

    random.seed(seeds[i+amount]+3920923)
    np.random.seed(seeds[i+amount]+19827)

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen=0

    k, r, Q_k_stored = RW_lambda_HP8(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,1] = len(bounds2) * np.log(T) + 2*negLL



    print(i, 'of', amount, 'done')



df_BIC1 = pd.DataFrame(BIC1)
df_BIC2 = pd.DataFrame(BIC2)

# Save the DataFrame to an Excel file
df_BIC1.to_excel(os.path.join(save_dir, '1HP8_BIC_eps-alpha-lambda.xlsx'))
df_BIC2.to_excel(os.path.join(save_dir, '2HP8_BIC_eps-alpha.xlsx'))













######## HP4 ADVERSARIAL CONTEXT ########
seeds = list(range(4*amount))  # Random seeds

BIC1 = np.zeros([amount,2])
BIC2 = np.zeros([amount,2])


### sim model 1: eps, alpha, unchosen ###
for i in range(amount):

    random.seed(seeds[i]+838743)
    np.random.seed(seeds[i]+21832)

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen = random.uniform(-3, 3)

    k, r, Q_k_stored = RW_lambda_HP4(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,1] = len(bounds2) * np.log(T) + 2*negLL




### sim model 2: eps, alpha ###

    random.seed(seeds[i+amount]+6970942)
    np.random.seed(seeds[i+amount]+583921)

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen=0

    k, r, Q_k_stored = RW_lambda_HP4(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,1] = len(bounds2) * np.log(T) + 2*negLL



    print(i, 'of', amount, 'done')



df_BIC1 = pd.DataFrame(BIC1)
df_BIC2 = pd.DataFrame(BIC2)

# Save the DataFrame to an Excel file
df_BIC1.to_excel(os.path.join(save_dir, '1HP4_BIC_eps-alpha-lambda.xlsx'))
df_BIC2.to_excel(os.path.join(save_dir, '2HP4_BIC_eps-alpha.xlsx'))







######## HP2 ADVERSARIAL CONTEXT ########
seeds = list(range(4*amount))  # Random seeds

BIC1 = np.zeros([amount,2])
BIC2 = np.zeros([amount,2])


### sim model 1: eps, alpha, unchosen ###
for i in range(amount):

    random.seed(seeds[i]+38758767)
    np.random.seed(seeds[i]+30988573)

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen = random.uniform(-3, 3)

    k, r, Q_k_stored = RW_lambda_HP2(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC1[i,1] = len(bounds2) * np.log(T) + 2*negLL




### sim model 2: eps, alpha ###

    random.seed(seeds[i+amount]+7483)
    np.random.seed(seeds[i+amount]+1234456)

    Q_alpha = random.uniform(0, 1)
    eps = random.uniform(0, 1)
    unchosen=0

    k, r, Q_k_stored = RW_lambda_HP2(Q_alpha=Q_alpha,eps=eps,unchosen=unchosen,T=T,Q_int=Q_int)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,0] = len(bounds1) * np.log(T) + 2*negLL


    #fit model 2: RW, fitting eps and alpha
    result = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL = result.fun
    param_fits = result.x

    BIC2[i,1] = len(bounds2) * np.log(T) + 2*negLL



    print(i, 'of', amount, 'done')



df_BIC1 = pd.DataFrame(BIC1)
df_BIC2 = pd.DataFrame(BIC2)

# Save the DataFrame to an Excel file
df_BIC1.to_excel(os.path.join(save_dir, '1HP2_BIC_eps-alpha-lambda.xlsx'))
df_BIC2.to_excel(os.path.join(save_dir, '2HP2_BIC_eps-alpha.xlsx'))