#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Meta-learning of lambda with policy-gradient method
Contexts:
Reinforced variability: reward dependent on how variable options are chosen according to
*Hide and Seek game with 8 choice alternatives
*Neuringer human/pigeon data 2 choice alternatives
*Neuringer human/pigeon data 4 choice alternatives
*Neuringer human/pigeon data 8 choice alternatives
*Jensen rat data 5 choice alternatives
            
Critical parameters: epsilon, learning rate, unchosen addition
This script will plot the evolution of optimal learning rate in five adversarial contexts for a fixed learning rate and fixed lambda

In this script, random seeds are used for each simulation.

@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random


#simulation of Rescorla-Wagner model with meta-learning of unchosen option value 
#meta-learning goes through parameter ML which is transformed to unchosen with a logit transformation
#rewards are baselined
def MLuc_HAS(Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated

    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
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
        #unchosen is calculated with a logit transformation of the ML
        uc = (2/K)*(np.exp(ML)/(1+np.exp(ML)))-(1/K)
        uc_mean = (2/K)*(np.exp(ML_mean)/(1+np.exp(ML_mean)))-(1/K)
        uc_std = (2/K)*(np.exp(ML_std)/(1+np.exp(ML_std)))-(1/K)
        
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
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

        #variable

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
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if ((t+1)%update)==0: #update every x=update amount of trials
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            baseline_reward = R_mean-av_reward_stored[begin+1]

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
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored





def MLuc_HP8(Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated

    
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.ones(K_seq)*20

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
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
        #unchosen is calculated with a logit transformation of the ML
        uc = (2/K)*(np.exp(ML)/(1+np.exp(ML)))-(1/K)
        uc_mean = (2/K)*(np.exp(ML_mean)/(1+np.exp(ML_mean)))-(1/K)
        uc_std = (2/K)*(np.exp(ML_std)/(1+np.exp(ML_std)))-(1/K)
        
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
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
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            baseline_reward = R_mean-av_reward_stored[begin+1]

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
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored


def MLuc_HP4(Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated

    
    K=4 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(4) for b in range(4) for c in range(4)])
    Freq1 = np.ones(K_seq)*20

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
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
        #unchosen is calculated with a logit transformation of the ML
        uc = (2/K)*(np.exp(ML)/(1+np.exp(ML)))-(1/K)
        uc_mean = (2/K)*(np.exp(ML_mean)/(1+np.exp(ML_mean)))-(1/K)
        uc_std = (2/K)*(np.exp(ML_std)/(1+np.exp(ML_std)))-(1/K)
        
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
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
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if ((t+1)%update)==0: #update every x=update amount of trials
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            baseline_reward = R_mean-av_reward_stored[begin+1]

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
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored



def MLuc_HP2(Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated

    
    K=2 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on ucilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a,b,c,d,e,f] for a in range(2) for b in range(2) for c in range(2) for d in range(2) for e in range(2) for f in range(2)])
    Freq1 = np.ones(K_seq)*20

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
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
        #unchosen is calculated with a logit transformation of the ML
        uc = (2/K)*(np.exp(ML)/(1+np.exp(ML)))-(1/K)
        uc_mean = (2/K)*(np.exp(ML_mean)/(1+np.exp(ML_mean)))-(1/K)
        uc_std = (2/K)*(np.exp(ML_std)/(1+np.exp(ML_std)))-(1/K)
        
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
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
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if ((t+1)%update)==0: #update every x=update amount of trials
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            baseline_reward = R_mean-av_reward_stored[begin+1]

            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 

            ### UPDATE MEAN IN NORMAL SPACE
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
           
            ### UPDATE LOG(MEAN)          
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)

            ### SAMPLE
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored


def MLuc_rat(Q_alpha, eps, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, update):
    #Q_alpha            --->        learning rate
    #eps                --->        epsilon; probability with which an action is chosen randomly
    #reward_alpha       --->        learning rate of baseline reward
    #ML_alpha_mean      --->        learning rate for mean of meta-learning parameter
    #ML_alpha_mean      --->        learning rate for std of meta-learning parameter
    #T                  --->        amount of trials for each simulation
    #K                  --->        amount of choice options
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #ML_mean_int        --->        initial value for the mean of unchosen
    #ML_std_int         --->        initial value for the standard deviation of unchosen
    #update             --->        this number equals the amount of trials after which the meta-learning parameter gets updated

    
    K=5 #the amount of choice options
    K_seq = 125 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on lrilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(5) for b in range(5) for c in range(5)])
    Freq1 = np.ones(K_seq)

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    uc_var = np.zeros((T), dtype=float)
    uc_var_mean = np.zeros((T), dtype=float)
    uc_var_std = np.zeros((T), dtype=float)
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
        #unchosen is calculated with a logit transformation of the ML
        uc = (2/K)*(np.exp(ML)/(1+np.exp(ML)))-(1/K)
        uc_mean = (2/K)*(np.exp(ML_mean)/(1+np.exp(ML_mean)))-(1/K)
        uc_std = (2/K)*(np.exp(ML_std)/(1+np.exp(ML_std)))-(1/K)
        
        uc_var[t] = uc
        uc_var_mean[t] = uc_mean
        uc_var_std[t] = uc_std
        # store values for Q and unchosen
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
        Q_k[np.arange(len(Q_k)) != k[t]] += uc

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            begin = t-update
            R_mean = np.mean(r[begin+1:t+1])
            baseline_reward = R_mean-av_reward_stored[begin+1]

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
            
  
    return k, r, Q_k_stored, uc_var, uc_var_mean, uc_var_std, ML_stored, ML_mean_stored, ML_std_stored


sim_nr = 1
amount_of_sim = 500
T = 10000

Q_alpha = 0.4
eps = 0.23


uc_mean_int = 0
MLuc_mean_int = uc_mean_int
ML_std_int = 1 

update = 10
Q_int = 1
reward_alpha = 0.25
ML_alpha_mean = 0.5
ML_alpha_std = 0.1 

seeds = list(range(5*amount_of_sim))  # Random seeds



#################################################################
#SIMULATIONS
#################################################################
#hide and seek
#for time plots:
g_varHAS = np.zeros([amount_of_sim, T])
g_mean_varHAS = np.zeros([amount_of_sim, T])
g_std_varHAS = np.zeros([amount_of_sim, T])

ML_varHAS = np.zeros([amount_of_sim, T])
ML_mean_varHAS = np.zeros([amount_of_sim, T])
ML_std_varHAS = np.zeros([amount_of_sim, T])
#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim])
    np.random.seed(seeds[sim])

    k, r, Q_k_stored, unchosen_varHAS, unchosen_varHAS_mean, unchosen_varHAS_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = MLuc_HAS(Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, update = 10)

    #for time plot:
    g_varHAS[sim,:] = unchosen_varHAS
    g_mean_varHAS[sim,:] = unchosen_varHAS_mean
    g_std_varHAS[sim,:] = unchosen_varHAS_std

    ML_varHAS[sim,:] = MLunchosen_stored
    ML_mean_varHAS[sim,:] = MLunchosen_mean_stored
    ML_std_varHAS[sim,:] = MLunchosen_std_stored
    
    print('check', sim, '/', amount_of_sim*5)

#for time plot:
varHAS_sampled_lambda_mean = np.mean(g_varHAS, axis=0)*8 #this is the mean of the sampled lambda
varHAS_sampled_lambda_std = np.std(g_varHAS, axis=0)*8 #this is the std of the sampled lambda
varHAS_sampled_lambda_ste = varHAS_sampled_lambda_std/(np.sqrt(amount_of_sim))

varHAS_mean_lambda_mean = np.mean(g_mean_varHAS, axis=0) #this is the mean of the sampled lambda
varHAS_mean_lambda_std = np.std(g_mean_varHAS, axis=0) #this is the std of the sampled lambda
varHAS_mean_lambda_ste = varHAS_mean_lambda_std/(np.sqrt(amount_of_sim))

varHAS_std_lambda_mean = np.mean(g_std_varHAS, axis=0) #this is the mean of the sampled lambda
varHAS_std_lambda_std = np.std(g_std_varHAS, axis=0) #this is the std of the sampled lambda


varHAS_sampled_ML_mean = np.mean(ML_varHAS, axis=0) #this is the mean of the sampled ML
varHAS_sampled_ML_std = np.std(ML_varHAS, axis=0) #this is the std of the sampled ML
varHAS_sampled_ML_ste = varHAS_sampled_ML_std/(np.sqrt(amount_of_sim))

varHAS_mean_ML_mean = np.mean(ML_mean_varHAS, axis=0) #this is the mean of the sampled ML
varHAS_mean_ML_std = np.std(ML_mean_varHAS, axis=0) #this is the std of the sampled ML
varHAS_mean_ML_ste = varHAS_mean_ML_std/(np.sqrt(amount_of_sim))

varHAS_std_ML_mean = np.mean(ML_std_varHAS, axis=0) #this is the mean of the sampled ML
varHAS_std_ML_std = np.std(ML_std_varHAS, axis=0) #this is the std of the sampled ML



#Human and pigeon - 8 choices
#for time plots:
g_varHP8 = np.zeros([amount_of_sim, T])
g_mean_varHP8 = np.zeros([amount_of_sim, T])
g_std_varHP8 = np.zeros([amount_of_sim, T])

ML_varHP8 = np.zeros([amount_of_sim, T])
ML_mean_varHP8 = np.zeros([amount_of_sim, T])
ML_std_varHP8 = np.zeros([amount_of_sim, T])
#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim+amount_of_sim])
    np.random.seed(seeds[sim+amount_of_sim]) 

    k, r, Q_k_stored, unchosen_varHP8, unchosen_varHP8_mean, unchosen_varHP8_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = MLuc_HP8(Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, update = 10)

    #for time plot:
    g_varHP8[sim,:] = unchosen_varHP8
    g_mean_varHP8[sim,:] = unchosen_varHP8_mean
    g_std_varHP8[sim,:] = unchosen_varHP8_std

    ML_varHP8[sim,:] = MLunchosen_stored
    ML_mean_varHP8[sim,:] = MLunchosen_mean_stored
    ML_std_varHP8[sim,:] = MLunchosen_std_stored
    
    print('check', sim+amount_of_sim, '/', amount_of_sim*5)

#for time plot:
varHP8_sampled_lambda_mean = np.mean(g_varHP8, axis=0)*8 #this is the mean of the sampled lambda
varHP8_sampled_lambda_std = np.std(g_varHP8, axis=0)*8 #this is the std of the sampled lambda
varHP8_sampled_lambda_ste = varHP8_sampled_lambda_std/(np.sqrt(amount_of_sim))

varHP8_mean_lambda_mean = np.mean(g_mean_varHP8, axis=0) #this is the mean of the sampled lambda
varHP8_mean_lambda_std = np.std(g_mean_varHP8, axis=0) #this is the std of the sampled lambda
varHP8_mean_lambda_ste = varHP8_mean_lambda_std/(np.sqrt(amount_of_sim))

varHP8_std_lambda_mean = np.mean(g_std_varHP8, axis=0) #this is the mean of the sampled lambda
varHP8_std_lambda_std = np.std(g_std_varHP8, axis=0) #this is the std of the sampled lambda


varHP8_sampled_ML_mean = np.mean(ML_varHP8, axis=0) #this is the mean of the sampled ML
varHP8_sampled_ML_std = np.std(ML_varHP8, axis=0) #this is the std of the sampled ML
varHP8_sampled_ML_ste = varHP8_sampled_ML_std/(np.sqrt(amount_of_sim))

varHP8_mean_ML_mean = np.mean(ML_mean_varHP8, axis=0) #this is the mean of the sampled ML
varHP8_mean_ML_std = np.std(ML_mean_varHP8, axis=0) #this is the std of the sampled ML
varHP8_mean_ML_ste = varHP8_mean_ML_std/(np.sqrt(amount_of_sim))

varHP8_std_ML_mean = np.mean(ML_std_varHP8, axis=0) #this is the mean of the sampled ML
varHP8_std_ML_std = np.std(ML_std_varHP8, axis=0) #this is the std of the sampled ML



#Human and pigeon 4 choices
#for time plots:
g_varHP4 = np.zeros([amount_of_sim, T])
g_mean_varHP4 = np.zeros([amount_of_sim, T])
g_std_varHP4 = np.zeros([amount_of_sim, T])

ML_varHP4 = np.zeros([amount_of_sim, T])
ML_mean_varHP4 = np.zeros([amount_of_sim, T])
ML_std_varHP4 = np.zeros([amount_of_sim, T])
#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim+2*amount_of_sim])
    np.random.seed(seeds[sim+2*amount_of_sim])

    k, r, Q_k_stored, unchosen_varHP4, unchosen_varHP4_mean, unchosen_varHP4_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = MLuc_HP4(Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, update = 10)

    #for time plot:
    g_varHP4[sim,:] = unchosen_varHP4
    g_mean_varHP4[sim,:] = unchosen_varHP4_mean
    g_std_varHP4[sim,:] = unchosen_varHP4_std

    ML_varHP4[sim,:] = MLunchosen_stored
    ML_mean_varHP4[sim,:] = MLunchosen_mean_stored
    ML_std_varHP4[sim,:] = MLunchosen_std_stored

    print('check', sim+amount_of_sim*2, '/', amount_of_sim*5)

#for time plot:
varHP4_sampled_lambda_mean = np.mean(g_varHP4, axis=0)*4 #this is the mean of the sampled lambda
varHP4_sampled_lambda_std = np.std(g_varHP4, axis=0)*4 #this is the std of the sampled lambda
varHP4_sampled_lambda_ste = varHP4_sampled_lambda_std/(np.sqrt(amount_of_sim))

varHP4_mean_lambda_mean = np.mean(g_mean_varHP4, axis=0) #this is the mean of the sampled lambda
varHP4_mean_lambda_std = np.std(g_mean_varHP4, axis=0) #this is the std of the sampled lambda
varHP4_mean_lambda_ste = varHP4_mean_lambda_std/(np.sqrt(amount_of_sim))

varHP4_std_lambda_mean = np.mean(g_std_varHP4, axis=0) #this is the mean of the sampled lambda
varHP4_std_lambda_std = np.std(g_std_varHP4, axis=0) #this is the std of the sampled lambda


varHP4_sampled_ML_mean = np.mean(ML_varHP4, axis=0) #this is the mean of the sampled ML
varHP4_sampled_ML_std = np.std(ML_varHP4, axis=0) #this is the std of the sampled ML
varHP4_sampled_ML_ste = varHP4_sampled_ML_std/(np.sqrt(amount_of_sim))

varHP4_mean_ML_mean = np.mean(ML_mean_varHP4, axis=0) #this is the mean of the sampled ML
varHP4_mean_ML_std = np.std(ML_mean_varHP4, axis=0) #this is the std of the sampled ML
varHP4_mean_ML_ste = varHP4_mean_ML_std/(np.sqrt(amount_of_sim))

varHP4_std_ML_mean = np.mean(ML_std_varHP4, axis=0) #this is the mean of the sampled ML
varHP4_std_ML_std = np.std(ML_std_varHP4, axis=0) #this is the std of the sampled ML




#human and pigeon 2 choices
#for time plots:
g_varHP2 = np.zeros([amount_of_sim, T])
g_mean_varHP2 = np.zeros([amount_of_sim, T])
g_std_varHP2 = np.zeros([amount_of_sim, T])

ML_varHP2 = np.zeros([amount_of_sim, T])
ML_mean_varHP2 = np.zeros([amount_of_sim, T])
ML_std_varHP2 = np.zeros([amount_of_sim, T])
#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim+3*amount_of_sim])
    np.random.seed(seeds[sim+3*amount_of_sim])

    k, r, Q_k_stored, unchosen_varHP2, unchosen_varHP2_mean, unchosen_varHP2_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = MLuc_HP2(Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, update = 10)

    #for time plot:
    g_varHP2[sim,:] = unchosen_varHP2
    g_mean_varHP2[sim,:] = unchosen_varHP2_mean
    g_std_varHP2[sim,:] = unchosen_varHP2_std

    ML_varHP2[sim,:] = MLunchosen_stored
    ML_mean_varHP2[sim,:] = MLunchosen_mean_stored
    ML_std_varHP2[sim,:] = MLunchosen_std_stored

    print('check', sim+amount_of_sim*3, '/', amount_of_sim*5)

#for time plot:
varHP2_sampled_lambda_mean = np.mean(g_varHP2, axis=0)*2 #this is the mean of the sampled lambda
varHP2_sampled_lambda_std = np.std(g_varHP2, axis=0)*2 #this is the std of the sampled lambda
varHP2_sampled_lambda_ste = varHP2_sampled_lambda_std/(np.sqrt(amount_of_sim))

varHP2_mean_lambda_mean = np.mean(g_mean_varHP2, axis=0) #this is the mean of the sampled lambda
varHP2_mean_lambda_std = np.std(g_mean_varHP2, axis=0) #this is the std of the sampled lambda
varHP2_mean_lambda_ste = varHP2_mean_lambda_std/(np.sqrt(amount_of_sim))

varHP2_std_lambda_mean = np.mean(g_std_varHP2, axis=0) #this is the mean of the sampled lambda
varHP2_std_lambda_std = np.std(g_std_varHP2, axis=0) #this is the std of the sampled lambda


varHP2_sampled_ML_mean = np.mean(ML_varHP2, axis=0) #this is the mean of the sampled ML
varHP2_sampled_ML_std = np.std(ML_varHP2, axis=0) #this is the std of the sampled ML
varHP2_sampled_ML_ste = varHP2_sampled_ML_std/(np.sqrt(amount_of_sim))

varHP2_mean_ML_mean = np.mean(ML_mean_varHP2, axis=0) #this is the mean of the sampled ML
varHP2_mean_ML_std = np.std(ML_mean_varHP2, axis=0) #this is the std of the sampled ML
varHP2_mean_ML_ste = varHP2_mean_ML_std/(np.sqrt(amount_of_sim))

varHP2_std_ML_mean = np.mean(ML_std_varHP2, axis=0) #this is the mean of the sampled ML
varHP2_std_ML_std = np.std(ML_std_varHP2, axis=0) #this is the std of the sampled ML







#rats
#for time plots:
g_RAT = np.zeros([amount_of_sim, T])
g_mean_RAT = np.zeros([amount_of_sim, T])
g_std_RAT = np.zeros([amount_of_sim, T])

ML_RAT = np.zeros([amount_of_sim, T])
ML_mean_RAT = np.zeros([amount_of_sim, T])
ML_std_RAT = np.zeros([amount_of_sim, T])
#simulation:
for sim in range(amount_of_sim):

    random.seed(seeds[sim+4*amount_of_sim])
    np.random.seed(seeds[sim+4*amount_of_sim])

    k, r, Q_k_stored, unchosen_RAT, unchosen_RAT_mean, unchosen_RAT_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored = MLuc_rat(Q_alpha=Q_alpha, eps=eps, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=MLuc_mean_int, ML_std_int=ML_std_int, update = 10)

    #for time plot:
    g_RAT[sim,:] = unchosen_RAT
    g_mean_RAT[sim,:] = unchosen_RAT_mean
    g_std_RAT[sim,:] = unchosen_RAT_std

    ML_RAT[sim,:] = MLunchosen_stored
    ML_mean_RAT[sim,:] = MLunchosen_mean_stored
    ML_std_RAT[sim,:] = MLunchosen_std_stored

    print('check', sim+amount_of_sim*4, '/', amount_of_sim*5)

#for time plot:
RAT_sampled_lambda_mean = np.mean(g_RAT, axis=0)*5 #this is the mean of the sampled lambda
RAT_sampled_lambda_std = np.std(g_RAT, axis=0)*5 #this is the std of the sampled lambda
RAT_sampled_lambda_ste = RAT_sampled_lambda_std/(np.sqrt(amount_of_sim))

RAT_mean_lambda_mean = np.mean(g_mean_RAT, axis=0) #this is the mean of the sampled lambda
RAT_mean_lambda_std = np.std(g_mean_RAT, axis=0) #this is the std of the sampled lambda
RAT_mean_lambda_ste = RAT_mean_lambda_std/(np.sqrt(amount_of_sim))

RAT_std_lambda_mean = np.mean(g_std_RAT, axis=0) #this is the mean of the sampled lambda
RAT_std_lambda_std = np.std(g_std_RAT, axis=0) #this is the std of the sampled lambda


RAT_sampled_ML_mean = np.mean(ML_RAT, axis=0) #this is the mean of the sampled ML
RAT_sampled_ML_std = np.std(ML_RAT, axis=0) #this is the std of the sampled ML
RAT_sampled_ML_ste = RAT_sampled_ML_std/(np.sqrt(amount_of_sim))

RAT_mean_ML_mean = np.mean(ML_mean_RAT, axis=0) #this is the mean of the sampled ML
RAT_mean_ML_std = np.std(ML_mean_RAT, axis=0) #this is the std of the sampled ML
RAT_mean_ML_ste = RAT_mean_ML_std/(np.sqrt(amount_of_sim))

RAT_std_ML_mean = np.mean(ML_std_RAT, axis=0) #this is the mean of the sampled ML
RAT_std_ML_std = np.std(ML_std_RAT, axis=0) #this is the std of the sampled ML





#################################################################
#PLOTTING
#################################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = "3ML_adversarials_lambda"
save_dir = os.path.join(script_dir, folder_name)


time = np.linspace(1, T, T, endpoint=True)


#Time plot for epsilon in each environment:
fig_name = os.path.join(save_dir, f'sim{sim_nr}_lambda_optimization')
fig, ax12 = plt.subplots(figsize=(20, 10))


epsilonvarHAS, = ax12.plot(time, varHAS_sampled_lambda_mean, label=f'simulation study - 8 options', color='forestgreen', linewidth=1)
ax12.fill_between(time, varHAS_sampled_lambda_mean - varHAS_sampled_lambda_ste, varHAS_sampled_lambda_mean + varHAS_sampled_lambda_ste, color='forestgreen', alpha=0.2)

epsilonvarHP8, = ax12.plot(time, varHP8_sampled_lambda_mean, label=f'Humans and pigeon - 8 options', color='#ddae48',linewidth=1)
ax12.fill_between(time, varHP8_sampled_lambda_mean - varHP8_sampled_lambda_ste, varHP8_sampled_lambda_mean + varHP8_sampled_lambda_ste, color='#ddae48', alpha=0.2)

epsilonvarHP4, = ax12.plot(time, varHP4_sampled_lambda_mean, label=f'Humans and pigeon - 4 options', color='#88661E',linewidth=1)
ax12.fill_between(time, varHP4_sampled_lambda_mean - varHP4_sampled_lambda_ste, varHP4_sampled_lambda_mean + varHP4_sampled_lambda_ste, color='#88661E', alpha=0.2)

epsilonvarHP2, = ax12.plot(time, varHP2_sampled_lambda_mean, label=f'Humans and pigeon - 2 options', color='#493710',linewidth=1)
ax12.fill_between(time, varHP2_sampled_lambda_mean - varHP2_sampled_lambda_ste, varHP2_sampled_lambda_mean + varHP2_sampled_lambda_ste, color='#493710', alpha=0.2)

epsilonvarRAT, = ax12.plot(time, RAT_sampled_lambda_mean, label=f'Rats - 5 options', color='grey',linewidth=1)
ax12.fill_between(time, RAT_sampled_lambda_mean - RAT_sampled_lambda_ste, RAT_sampled_lambda_mean + RAT_sampled_lambda_ste, color='grey', alpha=0.2)

ax12.legend(handles=[epsilonvarHAS, epsilonvarHP8, epsilonvarHP4, epsilonvarHP2, epsilonvarRAT])



ax12.set_xlabel('trials', fontsize=16)
ax12.set_ylabel('lambda', fontsize=16)
plt.xlim([0, T])
plt.ylim([-1, 1])
plt.yticks(fontsize=15)
plt.xticks(fontsize = 15)
#ax12.legend(handles=[epsilonsta,epsilonvol,epsilonvar])
#ax12.set_title(f'meta-learning of epsilon, based on {amount_of_sim} simulations')
plt.savefig(fig_name)