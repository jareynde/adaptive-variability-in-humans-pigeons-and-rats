#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Meta-learning of learning rate with policy-gradient method
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
First reward schedule test:
    stable: [0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.3]
    volatile: [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1]
    adversarial: least frequent 60% of sequences
            
Critical parameters: epsilon, learning rate, unchosen addition (there is no chosen here)
This script will plot the evolution of optimal learning rate in the three contexts for a fixed learning rate and fixed lambda

In this script, random seeds are used for each simulation.

@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random
import matplotlib


matplotlib.rcParams['font.family'] = 'times new roman'
fontsize = 30 
##############################################################################################################
#Function of Rescorla-Wagner model with meta-learning of epsilon 
#Meta-learning goes through parameter ML which is transformed to epsilon with a logit transformation
#Rewards are baselined
##############################################################################################################
#ADVERSARIAL ENVIRONMENT
def simulate_RW_MLlr_adversarial(eps, unchosen, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update):
    #eps                --->        epsilon; probability to choose randomly
    #unchosen           --->        the added value to unchosen option values
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of lr
    #ML_std_int         --->        initial value for the standard deviation of lr
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on lrilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    lr_var = np.zeros((T), dtype=float)
    lr_var_mean = np.zeros((T), dtype=float)
    lr_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop
    

    for t in range(T):
        #learning rate is calculated with a logit transformation of the ML
        lr = np.exp(ML)/(1+np.exp(ML))
        lr_mean = np.exp(ML_mean)/(1+np.exp(ML_mean))
        lr_std = np.exp(ML_std)/(1+np.exp(ML_std))
        lr_var[t] = lr
        lr_var_mean[t] = lr_mean
        lr_var_std[t] = lr_std
        # store values for Q and learning rate
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

   
        # Adversarial environment
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984



        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if ((t+1)%update)==0: #update every x=update amount of trials
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            if (t+1) == update:
                baseline_reward = R_mean-av_reward_stored[begin+1]
            else:
                baseline_reward = R_mean-av_reward_stored[begin]

            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 

            ### UPDATE MEAN IN NORMAL SPACE
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean

            ### UPDATE LOG(STD) 
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)

            ### SAMPLE
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
    return k, r, Q_k_stored, lr_var, lr_var_mean, lr_var_std, ML_stored, ML_mean_stored, ML_std_stored


#STABLE ENVIRONMENT
def simulate_RW_MLlr_stable(eps, unchosen, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update, reward_stable):
    #eps                --->        epsilon; probability to choose randomly
    #unchosen           --->        the added value to unchosen option values
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of lr
    #ML_std_int         --->        initial value for the standard deviation of lr
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #reward_stable      --->        reward probabilities for 8 options
    
    K=8 #the amount of choice options
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on lrilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    lr_var = np.zeros((T), dtype=float)
    lr_var_mean = np.zeros((T), dtype=float)
    lr_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop
     

    for t in range(T):
        #learning rate is calculated with a logit transformation of the ML
        lr = np.exp(ML)/(1+np.exp(ML))
        lr_mean = np.exp(ML_mean)/(1+np.exp(ML_mean))
        lr_std = np.exp(ML_std)/(1+np.exp(ML_std))
        lr_var[t] = lr
        lr_var_mean[t] = lr_mean
        lr_var_std[t] = lr_std
        # store values for Q and learning rate
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

        #stable
        a1 = reward_stable[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])
        
        
        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k

        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if ((t+1)%update)==0: #update every x=update amount of trials
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            if (t+1) == update:
                baseline_reward = R_mean-av_reward_stored[begin+1]
            else:
                baseline_reward = R_mean-av_reward_stored[begin]
            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN IN NORMAL SPACE
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
        
            ### UPDATE LOG(STD) 
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)

            ### SAMPLE
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, lr_var, lr_var_mean, lr_var_std, ML_stored, ML_mean_stored, ML_std_stored


#VOLATILE ENVIRONMENT
def simulate_RW_MLlr_volatile(eps, unchosen, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update, reward_volatile):
    #eps                --->        epsilon; probability to choose randomly
    #unchosen           --->        the added value to unchosen option values
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of lr
    #ML_std_int         --->        initial value for the standard deviation of lr
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated
    #reward_volatile    --->        reward probabilities for 8 options (shuffles among options)
    
    K=8 #the amount of choice options
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on lrilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    lr_var = np.zeros((T), dtype=float)
    lr_var_mean = np.zeros((T), dtype=float)
    lr_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000]    

    for t in range(T):
        #learning rate is calculated with a logit transformation of the ML
        lr = np.exp(ML)/(1+np.exp(ML))
        lr_mean = np.exp(ML_mean)/(1+np.exp(ML_mean))
        lr_std = np.exp(ML_std)/(1+np.exp(ML_std))
        lr_var[t] = lr
        lr_var_mean[t] = lr_mean
        lr_var_std[t] = lr_std
        # store values for Q and learning rate
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

        
        # volatile
        if t in v:
            min_vol = np.min(reward_volatile)
            max_vol = np.max(reward_volatile)
            index_vol = np.where(reward_volatile == min_vol)[0]
            random.shuffle(index_vol)
            new_max_index = index_vol[0:3]
            for i in range(8):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])
        
        
        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k

        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if ((t+1)%update)==0: #update every x=update amount of trials
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            if (t+1) == update:
                baseline_reward = R_mean-av_reward_stored[begin+1]
            else:
                baseline_reward = R_mean-av_reward_stored[begin]
            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN IN NORMAL SPACE
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            
            ### UPDATE LOG(STD) 
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
           
            ### SAMPLE
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, lr_var, lr_var_mean, lr_var_std, ML_stored, ML_mean_stored, ML_std_stored



sim_nr = 1
amount_of_sim = 500
T = 10000
K = 8

reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]

eps = 0.23
unchosen = 0


lr_mean_int = 0.4
MLlr_mean_int = np.log(lr_mean_int/(1-lr_mean_int))
ML_std_int = 1

update = 10
Q_int = 1
reward_alpha = 0.25
ML_alpha_mean = 0.5
ML_alpha_std = 0.1 

seeds = list(range(3*amount_of_sim))  # Random seeds


#################################################################
#Simulations
#################################################################
# META-LEARNING OF LEARNING RATE IN A VARIABLE CONTEXT (VARIABILITY IS REINFORCED)
#for time plots:
lr_var = np.zeros([amount_of_sim, T])
lr_mean_var = np.zeros([amount_of_sim, T])
lr_std_var = np.zeros([amount_of_sim, T])

ML_var = np.zeros([amount_of_sim, T])
ML_mean_var = np.zeros([amount_of_sim, T])
ML_std_var = np.zeros([amount_of_sim, T])
#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim])
    np.random.seed(seeds[sim])

    k, r, Q_k_stored, LR_var, LR_var_mean, LR_var_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLlr_adversarial(eps=eps, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLlr_mean_int, ML_std_int=ML_std_int, update = 10)
    
    #for time plot
    lr_var[sim,:] = LR_var
    lr_mean_var[sim,:] = LR_var_mean
    lr_std_var[sim,:] = LR_var_std

    ML_var[sim,:] = ML_stored
    ML_mean_var[sim,:] = ML_mean_stored
    ML_std_var[sim,:] = ML_std_stored

    print('check',sim)



#for time plot:
var_sampled_LR_mean = np.mean(lr_var, axis=0) #this is the mean of the sampled eps
var_sampled_LR_std = np.std(lr_var, axis=0) #this is the std of the sampled eps
var_sampled_LR_ste = var_sampled_LR_std/(np.sqrt(amount_of_sim))

var_mean_LR_mean = np.mean(lr_mean_var, axis=0) #this is the mean of the sampled eps
var_mean_LR_std = np.std(lr_mean_var, axis=0) #this is the std of the sampled eps
var_mean_LR_ste = var_mean_LR_std/(np.sqrt(amount_of_sim))

var_std_LR_mean = np.mean(lr_std_var, axis=0) #this is the mean of the sampled eps
var_std_LR_std = np.std(lr_std_var, axis=0) #this is the std of the sampled eps



var_sampled_ML_mean = np.mean(ML_var, axis=0) #this is the mean of the sampled ML
var_sampled_ML_std = np.std(ML_var, axis=0) #this is the std of the sampled ML
var_sampled_ML_ste = var_sampled_ML_std/(np.sqrt(amount_of_sim))

var_mean_ML_mean = np.mean(ML_mean_var, axis=0) #this is the mean of the sampled ML
var_mean_ML_std = np.std(ML_mean_var, axis=0) #this is the std of the sampled ML
var_mean_ML_ste = var_mean_ML_std/(np.sqrt(amount_of_sim))

var_std_ML_mean = np.mean(ML_std_var, axis=0) #this is the mean of the sampled ML
var_std_ML_std = np.std(ML_std_var, axis=0) #this is the std of the sampled ML





# META-LEARNING OF EPSILON IN A STABLE CONTEXT (REWARD PROBABILITIES ARE CONSTANT)
#for time plots:
lr_sta = np.zeros([amount_of_sim, T])
lr_mean_sta = np.zeros([amount_of_sim, T])
lr_std_sta = np.zeros([amount_of_sim, T])

ML_sta = np.zeros([amount_of_sim, T])
ML_mean_sta = np.zeros([amount_of_sim, T])
ML_std_sta = np.zeros([amount_of_sim, T])


#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim+amount_of_sim])
    np.random.seed(seeds[sim+amount_of_sim])


    k, r, Q_k_stored, LR_sta, LR_sta_mean, LR_sta_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLlr_stable(eps=eps, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLlr_mean_int, ML_std_int=ML_std_int, update = 10, reward_stable=reward_stable)

    #for time plot:
    lr_sta[sim,:] = LR_sta 
    lr_mean_sta[sim,:]  = LR_sta_mean
    lr_std_sta[sim,:]  = LR_sta_std

    ML_sta[sim,:] = ML_stored 
    ML_mean_sta[sim,:]  = ML_mean_stored
    ML_std_sta[sim,:]  = ML_std_stored

    print('check',sim+amount_of_sim)


#for time plots:
sta_sampled_LR_mean = np.mean(lr_sta, axis=0) #this is the mean of the sampled eps
sta_sampled_LR_std = np.std(lr_sta, axis=0) #this is the std of the sampled eps
sta_sampled_LR_ste = sta_sampled_LR_std/(np.sqrt(amount_of_sim))

sta_mean_LR_mean = np.mean(lr_mean_sta, axis=0) #this is the mean of the sampled eps
sta_mean_LR_std = np.std(lr_mean_sta, axis=0) #this is the std of the sampled eps
sta_mean_LR_ste = sta_mean_LR_std/(np.sqrt(amount_of_sim))

sta_std_LR_mean = np.mean(lr_std_sta, axis=0) #this is the mean of the sampled eps
sta_std_LR_std = np.std(lr_std_sta, axis=0) #this is the std of the sampled eps




sta_sampled_ML_mean = np.mean(ML_sta, axis=0) #this is the mean of the sampled ML
sta_sampled_ML_std = np.std(ML_sta, axis=0) #this is the std of the sampled ML
sta_sampled_ML_ste = sta_sampled_ML_std/(np.sqrt(amount_of_sim))

sta_mean_ML_mean = np.mean(ML_mean_sta, axis=0) #this is the mean of the sampled ML
sta_mean_ML_std = np.std(ML_mean_sta, axis=0) #this is the std of the sampled ML
sta_mean_ML_ste = sta_mean_ML_std/(np.sqrt(amount_of_sim))

sta_std_ML_mean = np.mean(ML_std_sta, axis=0) #this is the mean of the sampled ML
sta_std_ML_std = np.std(ML_std_sta, axis=0) #this is the std of the sampled ML





# META-LEARNING OF EPSILON IN A VOLATILE CONTEXT (REWARD PROBABILITIES SHUFFLE)
#for time plots:
lr_vol = np.zeros([amount_of_sim, T])
lr_mean_vol = np.zeros([amount_of_sim, T])
lr_std_vol = np.zeros([amount_of_sim, T])

ML_vol = np.zeros([amount_of_sim, T])
ML_mean_vol = np.zeros([amount_of_sim, T])
ML_std_vol = np.zeros([amount_of_sim, T])


for sim in range(amount_of_sim):

    random.seed(seeds[sim+2*amount_of_sim])
    np.random.seed(seeds[sim+2*amount_of_sim])


    k, r, Q_k_stored, LR_vol, LR_vol_mean, LR_vol_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLlr_volatile(eps=eps, unchosen=unchosen, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLlr_mean_int, ML_std_int=ML_std_int, update = 10, reward_volatile=reward_volatile)
    

    #for time plot:
    lr_vol[sim,:] = LR_vol
    lr_mean_vol[sim,:] = LR_vol_mean
    lr_std_vol[sim,:] = LR_vol_std

    ML_vol[sim,:] = ML_stored
    ML_mean_vol[sim,:] = ML_mean_stored
    ML_std_vol[sim,:] = ML_std_stored

    print('check',sim+2*amount_of_sim)


#for time plots:
vol_sampled_LR_mean = np.mean(lr_vol, axis=0) #this is the mean of the sampled eps
vol_sampled_LR_std = np.std(lr_vol, axis=0) #this is the std of the sampled eps
vol_sampled_LR_ste = vol_sampled_LR_std/(np.sqrt(amount_of_sim))

vol_mean_LR_mean = np.mean(lr_mean_vol, axis=0) #this is the mean of the sampled eps
vol_mean_LR_std = np.std(lr_mean_vol, axis=0) #this is the std of the sampled eps
vol_mean_LR_ste = vol_mean_LR_std/(np.sqrt(amount_of_sim))

vol_std_LR_mean = np.mean(lr_std_vol, axis=0) #this is the mean of the sampled eps
vol_std_LR_std = np.std(lr_std_vol, axis=0) #this is the std of the sampled eps


vol_sampled_ML_mean = np.mean(ML_vol, axis=0) #this is the mean of the sampled ML
vol_sampled_ML_std = np.std(ML_vol, axis=0) #this is the std of the sampled ML
vol_sampled_ML_ste = vol_sampled_ML_std/(np.sqrt(amount_of_sim))

vol_mean_ML_mean = np.mean(ML_mean_vol, axis=0) #this is the mean of the sampled ML
vol_mean_ML_std = np.std(ML_mean_vol, axis=0) #this is the std of the sampled ML
vol_mean_ML_ste = vol_mean_ML_std/(np.sqrt(amount_of_sim))

vol_std_ML_mean = np.mean(ML_std_vol, axis=0) #this is the mean of the sampled ML
vol_std_ML_std = np.std(ML_std_vol, axis=0) #this is the std of the sampled ML




#################################################################
#Plotting + saving data
#################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = "1ML_lr/final"
save_dir = os.path.join(script_dir, folder_name)

time = np.linspace(1, T, T, endpoint=True)


#Time plot for learning rate in each environment:
fig_name = os.path.join(save_dir, f'sim{sim_nr}_MLlr_optimization')
fig, ax12 = plt.subplots(figsize=(20, 10))
epsilonsta, = ax12.plot(time, sta_sampled_LR_mean, label=f'stable environment', color = 'darkcyan')
ax12.fill_between(time, sta_sampled_LR_mean - sta_sampled_LR_ste, sta_sampled_LR_mean + sta_sampled_LR_ste, color='darkcyan', alpha=0.2)

epsilonvol, = ax12.plot(time, vol_sampled_LR_mean, label=f'volatile environment', color = 'darkorange')
ax12.fill_between(time, vol_sampled_LR_mean - vol_sampled_LR_ste, vol_sampled_LR_mean + vol_sampled_LR_ste, color='darkorange', alpha=0.2)

epsilonvar, = ax12.plot(time, var_sampled_LR_mean, label=f'adversarial environment', color = 'forestgreen')
ax12.fill_between(time, var_sampled_LR_mean - var_sampled_LR_ste, var_sampled_LR_mean + var_sampled_LR_ste, color='forestgreen', alpha=0.2)

ax12.set_xlabel('trials', fontsize=fontsize)
ax12.set_ylabel(r'$\alpha$ - learning rate', fontsize=fontsize)
plt.ylim([0, 1])
plt.xlim([0, T])
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
#ax12.legend(handles=[epsilonsta,epsilonvol,epsilonvar])
#ax12.set_title(f'meta-learning of epsilon, based on {amount_of_sim} simulations')
plt.savefig(fig_name)

#Time plot for learning rate in each environment:
fig_name = os.path.join(save_dir, f'sim{sim_nr}_ML_optimization')
fig, ax12 = plt.subplots(figsize=(20, 10))
epsilonsta, = ax12.plot(time, sta_mean_ML_mean, label=f'stable environment', color = 'darkcyan')
ax12.fill_between(time, sta_mean_ML_mean - sta_std_ML_mean, sta_mean_ML_mean + sta_std_ML_mean, color='darkcyan', alpha=0.2)

epsilonvol, = ax12.plot(time, vol_mean_ML_mean, label=f'volatile environment', color = 'darkorange')
ax12.fill_between(time, vol_mean_ML_mean - vol_std_ML_mean, vol_mean_ML_mean + vol_std_ML_mean, color='darkorange', alpha=0.2)

epsilonvar, = ax12.plot(time, var_mean_ML_mean, label=f'adversarial environment', color = 'forestgreen')
ax12.fill_between(time, var_mean_ML_mean - var_std_ML_mean, var_mean_ML_mean + var_std_ML_mean, color='forestgreen', alpha=0.2)

ax12.set_xlabel('trials', fontsize=fontsize)
ax12.set_ylabel('X', fontsize=fontsize)
plt.xlim([0, T])
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
#ax12.legend(handles=[epsilonsta,epsilonvol,epsilonvar])
#ax12.set_title(f'meta-learning of epsilon, based on {amount_of_sim} simulations')
plt.savefig(fig_name)



opt_lr_var = np.mean(var_mean_LR_mean[-100:])
opt_lr_sta = np.mean(sta_mean_LR_mean[-100:])
opt_lr_vol = np.mean(vol_mean_LR_mean[-100:])

print('optimal is', opt_lr_var, opt_lr_sta, opt_lr_vol)





df_lr_var = pd.DataFrame(lr_var)
df_lr_sta = pd.DataFrame(lr_sta)
df_lr_vol = pd.DataFrame(lr_vol)

df_lr_var.to_excel(os.path.join(save_dir, 'LR_sampled_var.xlsx'))
df_lr_sta.to_excel(os.path.join(save_dir, 'LR_sampled_sta.xlsx'))
df_lr_vol.to_excel(os.path.join(save_dir, 'LR_sampled_vol.xlsx'))



df_lr_mean_var = pd.DataFrame(lr_mean_var)
df_lr_mean_sta = pd.DataFrame(lr_mean_sta)
df_lr_mean_vol = pd.DataFrame(lr_mean_vol)

df_lr_mean_var.to_excel(os.path.join(save_dir, 'LR_mean_var.xlsx'))
df_lr_mean_sta.to_excel(os.path.join(save_dir, 'LR_mean_sta.xlsx'))
df_lr_mean_vol.to_excel(os.path.join(save_dir, 'LR_mean_vol.xlsx'))



df_lr_std_var = pd.DataFrame(lr_std_var)
df_lr_std_sta = pd.DataFrame(lr_std_sta)
df_lr_std_vol = pd.DataFrame(lr_std_vol)

df_lr_std_var.to_excel(os.path.join(save_dir, 'LR_std_var.xlsx'))
df_lr_std_sta.to_excel(os.path.join(save_dir, 'LR_std_sta.xlsx'))
df_lr_std_vol.to_excel(os.path.join(save_dir, 'LR_std_vol.xlsx'))





df_ML_var = pd.DataFrame(ML_var)
df_ML_sta = pd.DataFrame(ML_sta)
df_ML_vol = pd.DataFrame(ML_vol)

df_ML_var.to_excel(os.path.join(save_dir, 'ML_sampled_var.xlsx'))
df_ML_sta.to_excel(os.path.join(save_dir, 'ML_sampled_sta.xlsx'))
df_ML_vol.to_excel(os.path.join(save_dir, 'ML_sampled_vol.xlsx'))


df_ML_mean_var = pd.DataFrame(ML_mean_var)
df_ML_mean_sta = pd.DataFrame(ML_mean_sta)
df_ML_mean_vol = pd.DataFrame(ML_mean_vol)

df_ML_mean_var.to_excel(os.path.join(save_dir, 'ML_mean_var.xlsx'))
df_ML_mean_sta.to_excel(os.path.join(save_dir, 'ML_mean_sta.xlsx'))
df_ML_mean_vol.to_excel(os.path.join(save_dir, 'ML_mean_vol.xlsx'))


df_ML_std_var = pd.DataFrame(ML_std_var)
df_ML_std_sta = pd.DataFrame(ML_std_sta)
df_ML_std_vol = pd.DataFrame(ML_std_vol)

df_ML_std_var.to_excel(os.path.join(save_dir, 'ML_std_var.xlsx'))
df_ML_std_sta.to_excel(os.path.join(save_dir, 'ML_std_sta.xlsx'))
df_ML_std_vol.to_excel(os.path.join(save_dir, 'ML_std_vol.xlsx'))





store_param_values = {
    'simulation number' : sim_nr,
    'amount of trials' : T,
    'amount of simulations' : amount_of_sim,
    'amount of choice options' : K,
    'amount of trials after which meta-learning parameters are updated' : update,
    'initial Q-value' : Q_int,
    'epsilon' : eps,
    'unchosen value-bias' : unchosen,
    'learning rate for the mean of the meta-learning parameter' : ML_alpha_mean,
    'learning rate for the std of the meta-learning parameter' : ML_alpha_std,
    'initial mean learning rate value' : lr_mean_int,
    'initial standard deviation of meta-learning parameter in logit transform' : ML_std_int,
    'optimal learning rate in adversarial' : opt_lr_var,
    'optimal learning rate in stable' : opt_lr_sta,
    'optimal learning rate in volatile' : opt_lr_vol
     }

title_excel = os.path.join(save_dir, f'sim{sim_nr}a_fixed_parameter_values.xlsx')
df = pd.DataFrame(data=store_param_values, index=[1])
df.to_excel(title_excel, index=False)


#plt.show()

