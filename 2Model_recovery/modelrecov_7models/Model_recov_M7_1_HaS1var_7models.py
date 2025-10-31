#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Context:
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
    adversarial: least frequent 60% of sequences


Model recovery of 9 models in adversarial context,
according to the block structure of HaS experiment 1 (meaning Q-values and freq reset every 200 trials):

*1: eps-lr
*2: eps-lr-uc
*3: eps-lr-gam

*4: eps-lr-uc-gam
*5: eps-lr-uc-ucs
*6: eps-lr-gam-gams

*7: eps-lr-uc-ucs/w-gam
*8: eps-lr-uc-gam-gams

*9: eps-lr-uc-ucs/w-gam-gams

wrote the sim functions and te negll functions
sim functions are the same as in 'sim_HaS1var_9models.py'
double check the functions, then write a model recovery 
@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                                  
import random
from random import uniform
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
import os
from csv import DictWriter
import pandas as pd


import sys
val = sys.argv[1:]
assert len(val) == 1
sim = int(val[0])

#sample van geschatte waarden

#Simulations
#simulation of Rescorla-Wagner model in an adversarial context

#Model 1
def RW_eps_lr_adversarial(eps, lr, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan) 
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)

        
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
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

        choice = k[t].astype(int)
        reward = r[t].astype(int)

         # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k


    return k, r, rand, Q_k_stored, delta_stored


#Model 2:
def RW_eps_lr_uc_adversarial(eps, lr, uc, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
        
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
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, rand, Q_k_stored, delta_stored


#Model 3.1:
def RW_eps_lr_gamr_adversarial(eps, lr, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #gamr        --->       gamma; prob to take recent option
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)


    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    delta_stored = np.full(T, np.nan) #store delta values for each action
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)

        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
  
        if t<buffer:
            # make choice based on choice probababilities
            rand[t] = np.random.choice(2, p=[1-eps,eps])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
        if t>=buffer:
            rand[t] = np.random.choice(3, p=[1-eps-gamr,eps,gamr])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
            if rand[t] == 2:
                k[t] = np.random.choice(k[t-buffer:t])
                
                    
        
        
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

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        

    return k, r, rand, Q_k_stored, delta_stored

#Model 3.2:
def RW_eps_lr_gamnr_adversarial(eps, lr, gamnr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #gamnr      --->        gamma; prob to take nonrecent option
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    


    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    delta_stored = np.full(T, np.nan) #store delta values for each action
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)

        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
  
        if t<buffer:
            # make choice based on choice probababilities
            rand[t] = np.random.choice(2, p=[1-eps,eps])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
        if t>=buffer:
            rand[t] = np.random.choice(3, p=[1-eps-gamnr,eps,gamnr])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
            if rand[t] == 2:
                options = [x for x in range(K) if x not in k[t-buffer:t]]
                k[t] = np.random.choice(options)
        
        
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

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

    

    return k, r, rand, Q_k_stored, delta_stored


#Model 4.1:
def RW_eps_lr_uc_gamr_adversarial(eps, lr, uc, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #gamr        --->       gamma; prob to take recent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    


    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)

        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              

        if t<buffer:
            # make choice based on choice probababilities
            rand[t] = np.random.choice(2, p=[1-eps,eps])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
        if t>=buffer:
            rand[t] = np.random.choice(3, p=[1-eps-gamr,eps,gamr])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
            if rand[t] == 2:
                k[t] = np.random.choice(k[t-buffer:t])
        
        
        
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

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, rand, Q_k_stored, delta_stored

#Model 4.2:
def RW_eps_lr_uc_gamnr_adversarial(eps, lr, uc, gamnr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #gamnr      --->        gamma; prob to take nonrecent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    

    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    delta_stored = np.full(T, np.nan) #store delta values for each action
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)

        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              

        if t<buffer:
            # make choice based on choice probababilities
            rand[t] = np.random.choice(2, p=[1-eps,eps])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
        if t>=buffer:
            rand[t] = np.random.choice(3, p=[1-eps-gamnr,eps,gamnr])
            if rand[t] == 0:
                k[t] = np.argmax(Q_k)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
            if rand[t] == 2:                
                options = [x for x in range(K) if x not in k[t-buffer:t]]
                k[t] = np.random.choice(options)
        
        
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

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced


    return k, r, rand, Q_k_stored, delta_stored


#Model 5:
def RW_eps_lr_uc_ucs_adversarial(eps, lr, uc, ucs, w, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        value-bias
    #ucs        --->        value-bias for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    delta_stored = np.full(T, np.nan)  #store delta values for each action

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    
    S_k_stored = np.full((T,K_seq), np.nan) 
    S_k = np.ones(K_seq)*Q_int

    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
       
        # store values for Q
        Q_k_stored[t,:] = Q_k  
        S_k_stored[t,:] = S_k 
            
     
        if t == 0:
            next_seq_options = np.zeros(K)
    
        # make choice based on choice probababilities
        combined_QandS_info = Q_k + w*next_seq_options
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(combined_QandS_info)
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

            # update Q_values of sequences
            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index] -= ucs

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced


    return k, r, rand, Q_k_stored, delta_stored, S_k_stored




#Model 7.1:
def RW_eps_lr_uc_ucs_gamr_adversarial(eps, lr, uc, ucs, w, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #ucs        --->        value-bias for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #gamr        --->        gamma; prob to take recent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    S_k_stored = np.full((T,K_seq), np.nan) 
    S_k = np.ones(K_seq)*Q_int

    delta_stored = np.full(T, np.nan)  #store delta values for each action

    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
       
        # store values for Q
        Q_k_stored[t,:] = Q_k  
        S_k_stored[t,:] = S_k 
            
     
        if t == 0:
            next_seq_options = np.zeros(K)
    
        # make choice based on choice probababilities
        combined_QandS_info = Q_k + w*next_seq_options
        if t<buffer:
            # make choice based on choice probababilities
            rand[t] = np.random.choice(2, p=[1-eps,eps])
            if rand[t] == 0:
                k[t] = np.argmax(combined_QandS_info)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
        if t>=buffer:
            rand[t] = np.random.choice(3, p=[1-eps-gamr,eps,gamr])
            if rand[t] == 0:
                k[t] = np.argmax(combined_QandS_info)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
            if rand[t] == 2:
                k[t] = np.random.choice(k[t-buffer:t])
               
                
        
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

            # update Q_values of sequences
            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index] -= ucs

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    

    return k, r, rand, Q_k_stored, delta_stored, S_k_stored


#Model 7.2:
def RW_eps_lr_uc_ucs_gamnr_adversarial(eps, lr, uc, ucs, w, gamnr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #ucs        --->        value-bias for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #gamnr        --->        gamma; prob to take nonrecent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    S_k_stored = np.full((T,K_seq), np.nan) 
    S_k = np.ones(K_seq)*Q_int

    delta_stored = np.full(T, np.nan)  #store delta values for each action

    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
       
        # store values for Q
        Q_k_stored[t,:] = Q_k  
        S_k_stored[t,:] = S_k 
            
     
        if t == 0:
            next_seq_options = np.zeros(K)
    
        # make choice based on choice probababilities
        combined_QandS_info = Q_k + w*next_seq_options
        if t<buffer:
            # make choice based on choice probababilities
            rand[t] = np.random.choice(2, p=[1-eps,eps])
            if rand[t] == 0:
                k[t] = np.argmax(combined_QandS_info)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
        if t>=buffer:
            rand[t] = np.random.choice(3, p=[1-eps-gamnr,eps,gamnr])
            if rand[t] == 0:
                k[t] = np.argmax(combined_QandS_info)
            if rand[t] == 1:
                k[t] = np.random.choice(range(K))
            if rand[t] == 2:
                options = [x for x in range(K) if x not in k[t-buffer:t]]
                k[t] = np.random.choice(options)
        
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

            # update Q_values of sequences
            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index] -= ucs

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced
    

    return k, r, rand, Q_k_stored, delta_stored, S_k_stored


# #negative loglikelihoods for each model
#model 1
def negll_RW_eps_lr(params, k, r):

    eps, lr = params
    Q_int = 1
    K = int(np.max(k)+1)
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

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

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice


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
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

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
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

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
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

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
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

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

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

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
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

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
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

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



#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params

#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params



K=8
bounds1=[(0,1), (0,1)]
bounds2=[(0,1), (0,1),(-1/K,1/K)]
bounds3=[(0,1), (0,1), (0,1)]
bounds4=[(0,1), (0,1), (-1/K,1/K), (0,1)]
bounds5=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,10)]
bounds7=[(0,1), (0,1), (-1/K,1/K), (-1/K,1/K), (0,10), (0,1)]


constraint3 = LinearConstraint([1,0,1], 0, 1)
constraint4 = LinearConstraint([1,0,0,1], 0, 1)

constraint7 = LinearConstraint([1, 0, 0, 0, 0, 1], 0, 1)





T=800
Q_int = 1




strategies = ['best1bin','best1exp','rand1exp','rand2bin','rand2exp','randtobest1bin','randtobest1exp','currenttobest1bin','currenttobest1exp','best2exp','best2bin']
strategy = 'randtobest1bin'



#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params

#    model 6.1  eps, lr, gamr, gamsr = params
#    model 6.2  eps, lr, gamnr, gamsnr = params

#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params

#    model 8.1  eps, lr, uc, gamr, gamsr = params
#    model 8.2  eps, lr, uc, gamnr, gamsnr = params

#    model 9.1  eps, lr, uc, ucs, w, gamr, gamsr = params
#    model 9.2  eps, lr, uc, ucs, w, gamnr, gamsnr = params
    
#########################################################################################
#simulate model 1 [eps-lr]
#########################################################################################

'''
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(script_dir , 'output_9models/M7_1')


M71sampled = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'ucs', 'w', 'gamr'])



#    model 1    eps, lr = params
#    model 2    eps, lr, uc = params

#    model 3.1  eps, lr, gamr = params
#    model 3.2  eps, lr, gamnr = params

#    model 4.1  eps, lr, uc, gamr = params
#    model 4.2  eps, lr, uc, gamnr = params

#    model 5    eps, lr, uc, ucs, w = params

#    model 6.1  eps, lr, gamr, gamsr = params
#    model 6.2  eps, lr, gamnr, gamsnr = params

#    model 7.1  eps, lr, uc, ucs, w, gamr = params
#    model 7.2  eps, lr, uc, ucs, w, gamnr = params

#    model 8.1  eps, lr, uc, gamr, gamsr = params
#    model 8.2  eps, lr, uc, gamnr, gamsnr = params

#    model 9.1  eps, lr, uc, ucs, w, gamr, gamsr = params
#    model 9.2  eps, lr, uc, ucs, w, gamnr, gamsnr = params

M71BIC_all = pd.DataFrame(index=range(0, 1), columns=['sim', 
                                                      'M1eps-lr', 
                                                     'M2eps-lr-uc', 
                                                     'M31eps-lr-gam', 'M32eps-lr-gam', 
                                                     'M41eps-lr-uc-gam', 'M42eps-lr-uc-gam', 
                                                     'M5eps-lr-uc-ucs',
                                                     'M61eps-lr-gam-gams', 'M62eps-lr-gam-gams',
                                                     'M71eps-lr-uc-ucs-gam', 'M72eps-lr-uc-ucs-gam',
                                                     'M81eps-lr-uc-gam-gams', 'M82eps-lr-uc-gam-gams',
                                                     'M91eps-lr-uc-ucs-gam-gams', 'M92eps-lr-uc-ucs-gam-gams'])


M71toM1 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'negLL', 'BIC'])
M71toM2 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'negLL', 'BIC'])

M71toM31 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'gamr', 'negLL', 'BIC'])
M71toM32 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'gamnr', 'negLL', 'BIC'])

M71toM41 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'gamr', 'negLL', 'BIC'])
M71toM42 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'gamnr', 'negLL', 'BIC'])

M71toM5 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'ucs', 'w', 'gam', 'negLL', 'BIC'])

M71toM61 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'gamr', 'gamsr', 'negLL', 'BIC'])
M71toM62 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'gamnr', 'gamsnr', 'negLL', 'BIC'])

M71toM71 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'ucs', 'w', 'gamr', 'negLL', 'BIC'])
M71toM72 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'ucs', 'w', 'gamnr', 'negLL', 'BIC'])

M71toM81 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'gamr', 'gamsr', 'negLL', 'BIC'])
M71toM82 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'gamnr', 'gamsnr', 'negLL', 'BIC'])

M71toM91 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'ucs', 'w', 'gamr', 'gamsr', 'negLL', 'BIC'])
M71toM92 = pd.DataFrame(index=range(0, 1), columns=['sim', 'eps', 'lr', 'uc', 'ucs', 'w', 'gamnr', 'gamsnr', 'negLL', 'BIC'])



title_excel = os.path.join(save_dir, f'M71BIC_all.csv')
M71BIC_all.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71sampled.csv')
M71sampled.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM1.csv')
M71toM1.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM2.csv')
M71toM2.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM31.csv')
M71toM31.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM32.csv')
M71toM32.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM41.csv')
M71toM41.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM42.csv')
M71toM42.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM5.csv')
M71toM5.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM61.csv')
M71toM61.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM62.csv')
M71toM62.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM71.csv')
M71toM71.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM72.csv')
M71toM72.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM81.csv')
M71toM81.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM82.csv')
M71toM82.to_csv(title_excel, index=False)

title_excel = os.path.join(save_dir, f'M71toM91.csv')
M71toM91.to_csv(title_excel, index=False)
title_excel = os.path.join(save_dir, f'M71toM92.csv')
M71toM92.to_csv(title_excel, index=False)


'''
buffer = 4
bufferseq = 8

#Get parameter values from estiamtes participants
df = pd.read_csv("parameter_values/M7_1_HaS1var_modelfit.csv") 

#min and max
eps_min = df["eps"].min()
eps_max = df["eps"].max()

lr_min = df["lr"].min()
lr_max = df["lr"].max()

uc_min = df["uc"].min()
uc_max = df["uc"].max()

ucs_min = df["ucs"].min()
ucs_max = df["ucs"].max()

w_min = df["w"].min()
w_max = df["w"].max()

gamr_min = df["gamr"].min()
gamr_max = df["gamr"].max()



gamr_min = 0.1 #no use for recovery if gamr is zero, it will be the same as eps-lr model
w_min = 0.1

#sample between the min and max
true_eps = random.uniform(eps_min, eps_max)
true_lr = random.uniform(lr_min, lr_max)
true_uc = random.uniform(uc_min, uc_max)
true_ucs = random.uniform(ucs_min, ucs_max)
true_w = random.uniform(w_min, w_max)
true_gamr = random.uniform(gamr_min, 1-true_eps)

'''

random_index = random.randint(0, len(df) - 1)
random_row = df.iloc[random_index]

true_eps = random_row["eps"]
true_lr = random_row["lr"]
true_uc = random_row["uc"]
true_ucs = random_row["ucs"]
true_w = random_row["w"]
true_gamr = random_row["gamr"]
'''


line_to_write = {"sim": sim, "eps": true_eps, "lr": true_lr, "uc": true_uc, "ucs": true_ucs, "w": true_w, "gamr": true_gamr}
with open("M7_1/M71sampled.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "ucs", "w", "gamr"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()

#simulate data from model 1
k, r, rand, Q_k_stored, delta_stored, S_k_stored = RW_eps_lr_uc_ucs_gamr_adversarial(eps=true_eps, lr=true_lr, uc=true_uc, ucs=true_ucs, w=true_w, gamr=true_gamr, buffer=buffer, T=T, Q_int=Q_int)
k = k.astype(int)
r = r.astype(int)

#model 1 [eps-lr]
#eps, lr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M1BIC = len(bounds1) * np.log(T) + 2*negLL


#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "negLL": negLL, "BIC": M1BIC}
with open("M7_1/M71toM1.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()


#model 2 [eps-lr-uc]
#eps, lr, uc = params
negLL = np.inf #initialize negative log likelihood

result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M2BIC = len(bounds2) * np.log(T) + 2*negLL


#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "uc" : param_fits[2], "negLL": negLL, "BIC": M2BIC}
with open("M7_1/M71toM2.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()




#model 3.1 [eps-lr-gamr]
#eps, lr, gamr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_gamr, bounds=bounds3, args=(k,r, buffer), strategy=strategy, constraints=constraint3)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M31BIC = len(bounds3) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "gamr": param_fits[2], "negLL": negLL, "BIC": M31BIC}
with open("M7_1/M71toM31.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "gamr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()


#model 3.2 [eps-lr-gamnr]
#eps, lr, gamnr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_gamnr, bounds=bounds3, args=(k,r, buffer), strategy=strategy, constraints=constraint3)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M32BIC = len(bounds3) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "gamnr": param_fits[2], "negLL": negLL, "BIC": M32BIC}
with open("M7_1/M71toM32.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "gamnr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()


#model 4.1 [eps-lr-uc-gamr]
#eps, lr, uc, gamr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_uc_gamr, bounds=bounds4, args=(k,r, buffer), strategy=strategy, constraints=constraint4)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M41BIC = len(bounds4) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "uc": param_fits[2], "gamr": param_fits[3], "negLL": negLL, "BIC": M41BIC}
with open("M7_1/M71toM41.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "gamr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()


#model 4.2 [eps-lr-uc-gamnr]
#eps, lr, uc, gamnr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_uc_gamnr, bounds=bounds4, args=(k,r, buffer), strategy=strategy, constraints=constraint4)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M42BIC = len(bounds4) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "uc": param_fits[2], "gamnr": param_fits[3], "negLL": negLL, "BIC": M42BIC}
with open("M7_1/M71toM42.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "gamnr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()



#model 5 [eps-lr-uc-ucs-w]
#eps, lr, uc, ucs, w = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_uc_ucs, bounds=bounds5, args=(k,r), strategy=strategy)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M5BIC = len(bounds5) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "uc": param_fits[2], "ucs": param_fits[3], "w": param_fits[4], "negLL": negLL, "BIC": M5BIC}
with open("M7_1/M71toM5.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "ucs", "w", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()




#model 7.1 [eps-lr-uc-ucs-w-gamr]
#eps, lr, uc, ucs, w, gamr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_uc_ucs_gamr, bounds=bounds7, args=(k,r, buffer), strategy=strategy, constraints=constraint7)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M71BIC = len(bounds7) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "uc": param_fits[2], "ucs": param_fits[3], "w": param_fits[4], "gamr": param_fits[5], "negLL": negLL, "BIC": M71BIC}
with open("M7_1/M71toM71.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "ucs", "w", "gamr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()


#model 7.2 [eps-lr-uc-ucs-w-gamnr]
#eps, lr, uc, ucs, w, gamnr = params
negLL = np.inf #initialize negative log likelihood
result = differential_evolution(negll_RW_eps_lr_uc_ucs_gamnr, bounds=bounds7, args=(k,r, buffer), strategy=strategy, constraints=constraint7)

#increasing maxiter and popsize doesn't make LR estimations better
negLL = result.fun
param_fits = result.x
M72BIC = len(bounds7) * np.log(T) + 2*negLL

#store in dataframe
line_to_write = {"sim": sim, "eps": param_fits[0], "lr": param_fits[1], "uc": param_fits[2], "ucs": param_fits[3], "w": param_fits[4], "gamnr": param_fits[5], "negLL": negLL, "BIC": M72BIC}
with open("M7_1/M71toM72.csv", 'a') as f_object:
    field_names = ["sim", "eps", "lr", "uc", "ucs", "w", "gamnr", "negLL", "BIC"]
    dictwriter_object = DictWriter(f_object, fieldnames = field_names)
    dictwriter_object.writerow(line_to_write)
    f_object.close()



