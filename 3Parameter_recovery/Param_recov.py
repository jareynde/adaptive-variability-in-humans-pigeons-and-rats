#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Model recovery
Model: eps, alpha, lambda
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
import math


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
#model 1
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


def calculate_rmse(actual_values, predicted_values):
    n = len(actual_values)
    squared_diffs = [(actual_values[i] - predicted_values[i]) ** 2 for i in range(n)]
    mean_squared_diff = sum(squared_diffs) / n
    rmse = math.sqrt(mean_squared_diff)
    return rmse
def calculate_Rsquared(y_true, y_pred):
    # Calculate the mean of observed values
    mean_y_true = np.mean(y_true)

    # Calculate the Sum of Squared Residuals (SSR)
    ssr = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))

    # Calculate the Total Sum of Squares (SST)
    sst = sum((y_true[i] - mean_y_true)**2 for i in range(len(y_true)))

    # Calculate R-squared
    Rsquared = 1 - (ssr / sst)
    return Rsquared 
#Third, simuate data


T=500
Q_int = 1
recov_amount = 1000
strategies = ['best1bin','best1exp','rand1exp','rand2bin','rand2exp','randtobest1bin','randtobest1exp','currenttobest1bin','currenttobest1exp','best2exp','best2bin']
strategies = ['randtobest1bin']

save_dir = os.path.dirname(os.path.abspath(__file__))

seeds = list(range(3*recov_amount))  # Random seeds

### RAT ###
for strategy in strategies:
    results = pd.DataFrame(index=range(0, 1), columns=['true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'])
    for count in range(recov_amount):
        random.seed(seeds[count])
        np.random.seed(seeds[count])
        true_alpha = np.random.rand()
        true_eps = np.random.rand()
        K=5
        true_unchosen = np.random.uniform(-1/K, 1/K)
        bounds=[(0,1), (0,1),(-1/K,1/K)]


        k, r, Q_k_stored = RW_rat_lambda_adversarial(Q_alpha = true_alpha, eps=true_eps, unchosen=true_unchosen, T=T, Q_int=Q_int)
        results.at[count, 'true_alpha'] = true_alpha
        results.at[count, 'true_epsilon'] = true_eps
        results.at[count, 'true_unchosen'] = true_unchosen


    #Fourth, do parameter recovery

        negLL = np.inf #initialize negative log likelihood

        #take several initial guesses for Q_alpha and epsilon and loop through them



        result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds, args=(k,r), strategy=strategy)
        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        
        BIC = len(bounds) * np.log(T) + 2*negLL


        #store in dataframe
        results.at[count, 'recov_alpha'] = param_fits[0]
        results.at[count, 'recov_epsilon'] = param_fits[1]
        results.at[count, 'negLL'] = negLL
        results.at[count, 'BIC'] = BIC
        results.at[count, 'recov_unchosen'] = param_fits[2]


        
        print('checkpoint rat', count)



    title_excel = os.path.join(save_dir, 'results/rat_results_eps_lr_lambda.xlsx')
    results.to_excel(title_excel, index=False)
    results = pd.read_excel(title_excel)

    #columns are 'true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'
    results = results.to_numpy()
    true_alpha = results[0:, 0]
    true_epsilon = results[0:, 1]
    recov_alpha = results[0:, 2]
    recov_epsilon = results[0:, 3]
    negLL = results[0:, 4]
    BIC = results[0:, 5]
    true_unchosen = results[0:,6]
    recov_unchosen = results[0:,7]




    RMSE_eps = calculate_rmse(true_epsilon,recov_epsilon)
    RMSE_LR = calculate_rmse(true_alpha, recov_alpha)
    RMSE_unchosen = calculate_rmse(true_unchosen, recov_unchosen)

    Rsquared_eps = calculate_Rsquared(true_epsilon,recov_epsilon)
    Rsquared_LR = calculate_Rsquared(true_alpha,recov_alpha)
    Rsquared_unchosen = calculate_Rsquared(true_unchosen,recov_unchosen)

    eps_pearson = stats.pearsonr(true_epsilon, recov_epsilon)
    alpha_pearson = stats.pearsonr(true_alpha, recov_alpha)
    unchosen_pearson = stats.pearsonr(true_unchosen, recov_unchosen)

    eps_spearman = stats.spearmanr(true_epsilon, recov_epsilon) 
    alpha_spearman = stats.spearmanr(true_alpha, recov_alpha)
    unchosen_spearman = stats.spearmanr(true_unchosen, recov_unchosen)


    store_correlations = {
        'RMSE_eps': RMSE_eps,
        'RMSE_LR' : RMSE_LR,
        'RMSE_unchosen' : RMSE_unchosen,

        'Rsquared_eps' : Rsquared_eps ,
        'Rsquared_LR' : Rsquared_LR  ,
        'Rsquared_unchosen' : Rsquared_unchosen ,

        'eps_pearson' : eps_pearson ,
        'alpha_pearson' : alpha_pearson ,
        'unchosen_pearson' : unchosen_pearson,

        'eps_spearman' : eps_spearman,
        'alpha_spearman' : alpha_spearman,
        'unchosen_spearman' : unchosen_spearman,

    
     }

    title_excel = os.path.join(save_dir, 'results/rat_correlation_values.xlsx')
    df = pd.DataFrame(data=store_correlations)
    df.to_excel(title_excel, index=False)

    fontsize = 12

'''
    fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(18,6))
    ax1.scatter(x=true_epsilon, y=recov_epsilon,s=2,c='black')
    ax1.set_xlabel('true epsilon', fontsize=fontsize)
    ax1.set_ylabel('recovered epsilon', fontsize=fontsize)
    ax2.scatter(x=true_alpha, y=recov_alpha,s=2,c='black')
    ax2.set_xlabel('true learning rate', fontsize=fontsize)
    ax2.set_ylabel('recovered learning rate', fontsize=fontsize)
    ax3.scatter(x=true_unchosen, y=recov_unchosen,s=2,c='black')
    ax3.set_xlabel('true lambda', fontsize=fontsize)
    ax3.set_ylabel('recovered lambda', fontsize=fontsize)
    fig_name1 = os.path.join(save_dir, 'rat_true-vs-recov_eps_lr_lambda')
    plt.show()
    plt.savefig(fig_name1)

'''




### HP8 ###
for strategy in strategies:
    results = pd.DataFrame(index=range(0, 1), columns=['true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'])
    for count in range(recov_amount):
        random.seed(seeds[count]+123943)
        np.random.seed(seeds[count]+42310)
        true_alpha = np.random.rand()
        true_eps = np.random.rand()
        K=8
        true_unchosen = np.random.uniform(-1/K, 1/K)
        bounds=[(0,1), (0,1),(-1/K,1/K)]

        k, r, Q_k_stored = RW_lambda_HP8(Q_alpha = true_alpha, eps=true_eps, unchosen=true_unchosen, T=T, Q_int=Q_int)
        results.at[count, 'true_alpha'] = true_alpha
        results.at[count, 'true_epsilon'] = true_eps
        results.at[count, 'true_unchosen'] = true_unchosen


    #Fourth, do parameter recovery

        negLL = np.inf #initialize negative log likelihood

        #take several initial guesses for Q_alpha and epsilon and loop through them



        result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds, args=(k,r), strategy=strategy)
        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        
        BIC = len(bounds) * np.log(T) + 2*negLL


        #store in dataframe
        results.at[count, 'recov_alpha'] = param_fits[0]
        results.at[count, 'recov_epsilon'] = param_fits[1]
        results.at[count, 'negLL'] = negLL
        results.at[count, 'BIC'] = BIC
        results.at[count, 'recov_unchosen'] = param_fits[2]


        
        print('checkpoint HP8', count)



    title_excel = os.path.join(save_dir, 'results/HP8_results_eps_lr_lambda.xlsx')
    results.to_excel(title_excel, index=False)
    results = pd.read_excel(title_excel)

    #columns are 'true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'
    results = results.to_numpy()
    true_alpha = results[0:, 0]
    true_epsilon = results[0:, 1]
    recov_alpha = results[0:, 2]
    recov_epsilon = results[0:, 3]
    negLL = results[0:, 4]
    BIC = results[0:, 5]
    true_unchosen = results[0:,6]
    recov_unchosen = results[0:,7]




    RMSE_eps = calculate_rmse(true_epsilon,recov_epsilon)
    RMSE_LR = calculate_rmse(true_alpha, recov_alpha)
    RMSE_unchosen = calculate_rmse(true_unchosen, recov_unchosen)

    Rsquared_eps = calculate_Rsquared(true_epsilon,recov_epsilon)
    Rsquared_LR = calculate_Rsquared(true_alpha,recov_alpha)
    Rsquared_unchosen = calculate_Rsquared(true_unchosen,recov_unchosen)

    eps_pearson = stats.pearsonr(true_epsilon, recov_epsilon)
    alpha_pearson = stats.pearsonr(true_alpha, recov_alpha)
    unchosen_pearson = stats.pearsonr(true_unchosen, recov_unchosen)

    eps_spearman = stats.spearmanr(true_epsilon, recov_epsilon) 
    alpha_spearman = stats.spearmanr(true_alpha, recov_alpha)
    unchosen_spearman = stats.spearmanr(true_unchosen, recov_unchosen)


    store_correlations = {
        'RMSE_eps': RMSE_eps,
        'RMSE_LR' : RMSE_LR,
        'RMSE_unchosen' : RMSE_unchosen,

        'Rsquared_eps' : Rsquared_eps ,
        'Rsquared_LR' : Rsquared_LR  ,
        'Rsquared_unchosen' : Rsquared_unchosen ,

        'eps_pearson' : eps_pearson ,
        'alpha_pearson' : alpha_pearson ,
        'unchosen_pearson' : unchosen_pearson,

        'eps_spearman' : eps_spearman,
        'alpha_spearman' : alpha_spearman,
        'unchosen_spearman' : unchosen_spearman,

    
     }

    title_excel = os.path.join(save_dir, 'results/HP8_correlation_values.xlsx')
    df = pd.DataFrame(data=store_correlations)
    df.to_excel(title_excel, index=False)

'''
    fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(18,6))
    ax1.scatter(x=true_epsilon, y=recov_epsilon,s=2,c='black')
    ax1.set_xlabel('true epsilon')
    ax1.set_ylabel('recovered epsilon')
    ax2.scatter(x=true_alpha, y=recov_alpha,s=2,c='black')
    ax2.set_xlabel('true learning rate')
    ax2.set_ylabel('recovered learning rate')
    ax3.scatter(x=true_unchosen, y=recov_unchosen,s=2,c='black')
    ax3.set_xlabel('true lambda')
    ax3.set_ylabel('recovered lambda')
    fig_name1 = os.path.join(save_dir, 'HP8_true-vs-recov_eps_lr_lambda')
    plt.show()
    plt.savefig(fig_name1)
'''




### HP4 ###
for strategy in strategies:
    results = pd.DataFrame(index=range(0, 1), columns=['true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'])
    for count in range(recov_amount):
        random.seed(seeds[count]+394910)
        np.random.seed(seeds[count]+23492)
        true_alpha = np.random.rand()
        true_eps = np.random.rand()
        K=4
        true_unchosen = np.random.uniform(-1/K, 1/K)
        bounds=[(0,1), (0,1),(-1/K,1/K)]

        k, r, Q_k_stored = RW_lambda_HP4(Q_alpha = true_alpha, eps=true_eps, unchosen=true_unchosen, T=T, Q_int=Q_int)
        results.at[count, 'true_alpha'] = true_alpha
        results.at[count, 'true_epsilon'] = true_eps
        results.at[count, 'true_unchosen'] = true_unchosen


    #Fourth, do parameter recovery

        negLL = np.inf #initialize negative log likelihood

        #take several initial guesses for Q_alpha and epsilon and loop through them



        result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds, args=(k,r), strategy=strategy)
        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        
        BIC = len(bounds) * np.log(T) + 2*negLL


        #store in dataframe
        results.at[count, 'recov_alpha'] = param_fits[0]
        results.at[count, 'recov_epsilon'] = param_fits[1]
        results.at[count, 'negLL'] = negLL
        results.at[count, 'BIC'] = BIC
        results.at[count, 'recov_unchosen'] = param_fits[2]


        
        print('checkpoint HP4', count)



    title_excel = os.path.join(save_dir, 'results/HP4_results_eps_lr_lambda.xlsx')
    results.to_excel(title_excel, index=False)
    results = pd.read_excel(title_excel)

    #columns are 'true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'
    results = results.to_numpy()
    true_alpha = results[0:, 0]
    true_epsilon = results[0:, 1]
    recov_alpha = results[0:, 2]
    recov_epsilon = results[0:, 3]
    negLL = results[0:, 4]
    BIC = results[0:, 5]
    true_unchosen = results[0:,6]
    recov_unchosen = results[0:,7]




    RMSE_eps = calculate_rmse(true_epsilon,recov_epsilon)
    RMSE_LR = calculate_rmse(true_alpha, recov_alpha)
    RMSE_unchosen = calculate_rmse(true_unchosen, recov_unchosen)

    Rsquared_eps = calculate_Rsquared(true_epsilon,recov_epsilon)
    Rsquared_LR = calculate_Rsquared(true_alpha,recov_alpha)
    Rsquared_unchosen = calculate_Rsquared(true_unchosen,recov_unchosen)

    eps_pearson = stats.pearsonr(true_epsilon, recov_epsilon)
    alpha_pearson = stats.pearsonr(true_alpha, recov_alpha)
    unchosen_pearson = stats.pearsonr(true_unchosen, recov_unchosen)

    eps_spearman = stats.spearmanr(true_epsilon, recov_epsilon) 
    alpha_spearman = stats.spearmanr(true_alpha, recov_alpha)
    unchosen_spearman = stats.spearmanr(true_unchosen, recov_unchosen)


    store_correlations = {
        'RMSE_eps': RMSE_eps,
        'RMSE_LR' : RMSE_LR,
        'RMSE_unchosen' : RMSE_unchosen,

        'Rsquared_eps' : Rsquared_eps ,
        'Rsquared_LR' : Rsquared_LR  ,
        'Rsquared_unchosen' : Rsquared_unchosen ,

        'eps_pearson' : eps_pearson ,
        'alpha_pearson' : alpha_pearson ,
        'unchosen_pearson' : unchosen_pearson,

        'eps_spearman' : eps_spearman,
        'alpha_spearman' : alpha_spearman,
        'unchosen_spearman' : unchosen_spearman,

    
     }

    title_excel = os.path.join(save_dir, 'results/HP4_correlation_values.xlsx')
    df = pd.DataFrame(data=store_correlations)
    df.to_excel(title_excel, index=False)


'''
    fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(18,6))
    ax1.scatter(x=true_epsilon, y=recov_epsilon,s=2,c='black')
    ax1.set_xlabel('true epsilon')
    ax1.set_ylabel('recovered epsilon')
    ax2.scatter(x=true_alpha, y=recov_alpha,s=2,c='black')
    ax2.set_xlabel('true learning rate')
    ax2.set_ylabel('recovered learning rate')
    ax3.scatter(x=true_unchosen, y=recov_unchosen,s=2,c='black')
    ax3.set_xlabel('true lambda')
    ax3.set_ylabel('recovered lambda')
    fig_name1 = os.path.join(save_dir, 'HP4_true-vs-recov_eps_lr_lambda')
    plt.show()
    plt.savefig(fig_name1)
'''



### HP2 ###
for strategy in strategies:
    results = pd.DataFrame(index=range(0, 1), columns=['true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'])
    for count in range(recov_amount):
        random.seed(seeds[count]+2942)
        np.random.seed(seeds[count]+5693092)
        true_alpha = np.random.rand()
        true_eps = np.random.rand()
        K=2
        true_unchosen = np.random.uniform(-1/K, 1/K)
        bounds=[(0,1), (0,1),(-1/K,1/K)]

        k, r, Q_k_stored = RW_lambda_HP2(Q_alpha = true_alpha, eps=true_eps, unchosen=true_unchosen, T=T, Q_int=Q_int)
        results.at[count, 'true_alpha'] = true_alpha
        results.at[count, 'true_epsilon'] = true_eps
        results.at[count, 'true_unchosen'] = true_unchosen


    #Fourth, do parameter recovery

        negLL = np.inf #initialize negative log likelihood

        #take several initial guesses for Q_alpha and epsilon and loop through them



        result = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds, args=(k,r), strategy=strategy)
        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        
        BIC = len(bounds) * np.log(T) + 2*negLL


        #store in dataframe
        results.at[count, 'recov_alpha'] = param_fits[0]
        results.at[count, 'recov_epsilon'] = param_fits[1]
        results.at[count, 'negLL'] = negLL
        results.at[count, 'BIC'] = BIC
        results.at[count, 'recov_unchosen'] = param_fits[2]


        
        print('checkpoint HP2', count)



    title_excel = os.path.join(save_dir, 'results/HP2_results_eps_lr_lambda.xlsx')
    results.to_excel(title_excel, index=False)
    results = pd.read_excel(title_excel)

    #columns are 'true_alpha','true_epsilon', 'recov_alpha', 'recov_epsilon', 'negLL', 'BIC', 'true_unchosen', 'recov_unchosen'
    results = results.to_numpy()
    true_alpha = results[0:, 0]
    true_epsilon = results[0:, 1]
    recov_alpha = results[0:, 2]
    recov_epsilon = results[0:, 3]
    negLL = results[0:, 4]
    BIC = results[0:, 5]
    true_unchosen = results[0:,6]
    recov_unchosen = results[0:,7]




    RMSE_eps = calculate_rmse(true_epsilon,recov_epsilon)
    RMSE_LR = calculate_rmse(true_alpha, recov_alpha)
    RMSE_unchosen = calculate_rmse(true_unchosen, recov_unchosen)

    Rsquared_eps = calculate_Rsquared(true_epsilon,recov_epsilon)
    Rsquared_LR = calculate_Rsquared(true_alpha,recov_alpha)
    Rsquared_unchosen = calculate_Rsquared(true_unchosen,recov_unchosen)

    eps_pearson = stats.pearsonr(true_epsilon, recov_epsilon)
    alpha_pearson = stats.pearsonr(true_alpha, recov_alpha)
    unchosen_pearson = stats.pearsonr(true_unchosen, recov_unchosen)

    eps_spearman = stats.spearmanr(true_epsilon, recov_epsilon) 
    alpha_spearman = stats.spearmanr(true_alpha, recov_alpha)
    unchosen_spearman = stats.spearmanr(true_unchosen, recov_unchosen)


    store_correlations = {
        'RMSE_eps': RMSE_eps,
        'RMSE_LR' : RMSE_LR,
        'RMSE_unchosen' : RMSE_unchosen,

        'Rsquared_eps' : Rsquared_eps ,
        'Rsquared_LR' : Rsquared_LR  ,
        'Rsquared_unchosen' : Rsquared_unchosen ,

        'eps_pearson' : eps_pearson ,
        'alpha_pearson' : alpha_pearson ,
        'unchosen_pearson' : unchosen_pearson,

        'eps_spearman' : eps_spearman,
        'alpha_spearman' : alpha_spearman,
        'unchosen_spearman' : unchosen_spearman,

    
     }

    title_excel = os.path.join(save_dir, 'results//HP2_correlation_values.xlsx')
    df = pd.DataFrame(data=store_correlations)
    df.to_excel(title_excel, index=False)


'''
    fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(18,6))
    ax1.scatter(x=true_epsilon, y=recov_epsilon,s=2,c='black')
    ax1.set_xlabel('true epsilon')
    ax1.set_ylabel('recovered epsilon')
    ax2.scatter(x=true_alpha, y=recov_alpha,s=2,c='black')
    ax2.set_xlabel('true learning rate')
    ax2.set_ylabel('recovered learning rate')
    ax3.scatter(x=true_unchosen, y=recov_unchosen,s=2,c='black')
    ax3.set_xlabel('true lambda')
    ax3.set_ylabel('recovered lambda')
    fig_name1 = os.path.join(save_dir, 'HP2_true-vs-recov_eps_lr_lambda')
    plt.show()
    plt.savefig(fig_name1)

'''