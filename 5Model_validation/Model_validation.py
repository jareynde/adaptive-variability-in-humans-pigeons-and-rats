#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model validation for winning model on human data

@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                                  
import random
from random import uniform
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt     # plotting
import matplotlib





#For human participants, model 7.1 gives the best fit

#Model 7.1:
def H2_RW_eps_lr_uc_ucs_gamr_adversarial(eps, lr, uc, ucs, w, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #ucs        --->        value-bias for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #gamr        --->        gamma; prob to take recent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=2 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a,b,c,d,e,f] for a in range(2) for b in range(2) for c in range(2) for d in range(2) for e in range(2) for f in range(2)])
    Freq1 = np.ones(K_seq)*20

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    S_k_stored = np.full((T,K*K), np.nan) 
    S_k = np.ones(K*K)*Q_int

    seq_options_stored = np.full((T,K), np.nan)

    delta_stored = np.full(T, np.nan)  #store delta values for each action

    for t in range(T):

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

        if t > 0:
            seq_options2 = np.array([[a, b] for a in range(2) for b in range(2)])
            current_seq2 = k[t-1:t+1]
            current_index2 = np.where(np.all(seq_options2==current_seq2,axis=1))[0]
            # update Q_values of sequences
            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index2] -= ucs

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options2==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        seq_options_stored[t,:] = next_seq_options

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    

    return k, r, rand, Q_k_stored, delta_stored, S_k_stored, seq_options_stored




#Model 7.1:
def H4_RW_eps_lr_uc_ucs_gamr_adversarial(eps, lr, uc, ucs, w, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #ucs        --->        value-bias for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #gamr        --->        gamma; prob to take recent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    
    K=4 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(4) for b in range(4) for c in range(4)])
    Freq1 = np.ones(K_seq)*20

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    S_k_stored = np.full((T,K*K), np.nan) 
    S_k = np.ones(K*K)*Q_int

    seq_options_stored = np.full((T,K), np.nan)

    delta_stored = np.full(T, np.nan)  #store delta values for each action

    for t in range(T):

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

        if t > 0:
            seq_options2 = np.array([[a, b] for a in range(4) for b in range(4)])
            current_seq2 = k[t-1:t+1]
            current_index2 = np.where(np.all(seq_options2==current_seq2,axis=1))[0]
            # update Q_values of sequences
            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index2] -= ucs

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options2==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        seq_options_stored[t,:] = next_seq_options

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    

    return k, r, rand, Q_k_stored, delta_stored, S_k_stored, seq_options_stored


#Model 7.1:
def H8_RW_eps_lr_uc_ucs_gamr_adversarial(eps, lr, uc, ucs, w, gamr, buffer, T, Q_int):
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
    Freq1 = np.ones(K_seq)*20

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    S_k_stored = np.full((T,K_seq), np.nan) 
    S_k = np.ones(K_seq)*Q_int

    seq_options_stored = np.full((T,K), np.nan)

    delta_stored = np.full(T, np.nan)  #store delta values for each action

    for t in range(T):

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
            if current_freq < 21.6:
                r[t] = 1
            else: r[t] = 0
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            if r[t] == 1:
                Freq1 = Freq1*0.984

        if t > 0:
            # update Q_values of sequences
            S_k += ucs  # Apply the unchosen bias to all Q-values
            S_k[current_index] -= ucs

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        seq_options_stored[t,:] = next_seq_options

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    

    return k, r, rand, Q_k_stored, delta_stored, S_k_stored, seq_options_stored



#Model 4.1:
def P2_RW_eps_lr_uc_gamr_adversarial(eps, lr, uc, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #gamr        --->       gamma; prob to take recent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=2 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
      

    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a,b,c,d,e,f] for a in range(2) for b in range(2) for c in range(2) for d in range(2) for e in range(2) for f in range(2)])
    Freq1 = np.ones(K_seq)*20

    Freq_all = Freq1

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
    for t in range(T):
        
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

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        Freq_all = np.vstack((Freq_all,Freq1))

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, rand, Q_k_stored, delta_stored, Freq_all




#Model 4.1:
def P4_RW_eps_lr_uc_gamr_adversarial(eps, lr, uc, gamr, buffer, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #gamr        --->       gamma; prob to take recent option
    #conv       --->        convert to take recent or nonrecent option with gamma
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    
    K=4 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    

    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(4) for b in range(4) for c in range(4)])
    Freq1 = np.ones(K_seq)*20

    Freq_all = Freq1

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
    for t in range(T):
        
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

        choice = k[t].astype(int)
        reward = r[t].astype(int)

        Freq_all = np.vstack((Freq_all,Freq1))

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, rand, Q_k_stored, delta_stored, Freq_all




#Model 4.1:
def P8_RW_eps_lr_uc_gamr_adversarial(eps, lr, uc, gamr, buffer, T, Q_int):
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
    Freq1 = np.ones(K_seq)*20
    Freq_all = Freq1

    #Q values
    Q_k_stored = np.full((T,K), np.nan)  #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
    for t in range(T):
        
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
            if current_freq < 21.6:
                r[t] = 1
            else: r[t] = 0
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            if r[t] == 1:
                Freq1 = Freq1*0.984
            
        choice = k[t].astype(int)
        reward = r[t].astype(int)

        Freq_all = np.vstack((Freq_all,Freq1))

        # update Q values for chosen option:
        delta_k = reward - Q_k[choice]
        Q_k[choice] = Q_k[choice] + lr * delta_k
        delta_stored[t] = delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[choice] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, rand, Q_k_stored, delta_stored, Freq_all




#Model 2:
def R5_RW_eps_lr_uc_adversarial(eps, lr, uc, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)


    K=5 #the amount of choice options
    K_seq = 125 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.full(T, np.nan)  #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.full(T, np.nan)  #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b, c] for a in range(5) for b in range(5) for c in range(5)])
    Freq1 = np.ones(K_seq)

    #Q values
    Q_k_stored = np.full((T,K), np.nan) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    delta_stored = np.full(T, np.nan)  #store delta values for each action
    
    
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



#Functions to calculate U-values
def count_pairs(sequence):
    counts = Counter(zip(sequence[:-1], sequence[1:]))
    pair_counts = list(counts.values())
    return pair_counts

def u_value(seq, alternatives, two_alternatives):
    unique_values, count1 = np.unique(seq, return_counts=True)

    b1 = alternatives
    amount = len(seq)
    prob1 = count1/amount
    u1 = -np.sum(prob1*np.log2(prob1))/np.log2(b1)
    
    count2 = count_pairs(seq)
    b2 = two_alternatives
    prob2 = np.divide(count2,amount-1)
    u2 = -(np.sum(prob2*np.log2(prob2))+u1)/np.log2(b2)

    return u1, u2



script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(script_dir)
data_dir = os.path.join(model_dir, '4Data_fit/HPC_fit7models')
human_dir = os.path.join(data_dir, 'Human_output')
pigeon_dir = os.path.join(data_dir, 'Pigeon_output')
rat_dir = os.path.join(data_dir, 'Rat_output/all')


human_file = os.path.join(human_dir, 'Human_M7_1.csv')
pigeon_file = os.path.join(pigeon_dir, 'Pigeon_M4_1.csv')
rat_file = os.path.join(rat_dir, 'Ratall_M2.csv')

human_data = pd.read_csv(human_file)
pigeon_data = pd.read_csv(pigeon_file)
rat_data = pd.read_csv(rat_file)

amount_of_sim = 10

#####################################################################################
#validation of human model
#####################################################################################
Heps = human_data['eps'].to_numpy()
Hlr = human_data['alpha'].to_numpy()
Huc = human_data['lambda'].to_numpy()
Hucs = human_data['lambda_s'].to_numpy()
Hw = human_data['w'].to_numpy()
Hgamr = human_data['gamr'].to_numpy()

trials = human_data['trials'].to_numpy()
choices = human_data['choices'].to_numpy()

r_human = np.zeros(len(Heps))
rstd_human = np.zeros(len(Heps))
u2_human = np.zeros(len(Heps))
u2std_human = np.zeros(len(Heps))
hav_r = np.zeros((amount_of_sim,len(Heps)))
hav_u2 = np.zeros((amount_of_sim, len(Heps)))

for i in range(len(Heps)):
    eps = Heps[i]
    lr = Hlr[i]
    uc = 0#Huc[i]
    ucs = Hucs[i]
    w = Hw[i]
    gamr = Hgamr[i]

    K = choices[i]
    buffer = K/2
    #make sure buffer is integer
    buffer = buffer.astype(int)
    T = trials[i]
    Q_int = 1

    for j in range(amount_of_sim):

        if K == 2:
            k, r, rand, Q_k_stored, delta_stored, S_k_stored, seq_options_stored = H2_RW_eps_lr_uc_ucs_gamr_adversarial(eps=eps, lr=lr, uc=uc, ucs=ucs, w=w, gamr=gamr, buffer=buffer, T=T, Q_int=Q_int)
            print('in loop of K = 2 and this is', i)

        if K == 4:
            k, r, rand, Q_k_stored, delta_stored, S_k_stored, seq_options_stored = H4_RW_eps_lr_uc_ucs_gamr_adversarial(eps=eps, lr=lr, uc=uc, ucs=ucs, w=w, gamr=gamr, buffer=buffer, T=T, Q_int=Q_int)
            print('in loop of K = 4 and this is', i)
        if K == 8:
            k, r, rand, Q_k_stored, delta_stored, S_k_stored, seq_options_stored = H8_RW_eps_lr_uc_ucs_gamr_adversarial(eps=eps, lr=lr, uc=uc, ucs=ucs, w=w, gamr=gamr, buffer=buffer, T=T, Q_int=Q_int)
            print('in loop of K = 8 and this is', i)
        
        hav_r[j,i] = np.mean(r)
        xo,hav_u2[j,i] = u_value(k[-500:], alternatives=K, two_alternatives=K*K)

    
    r_human[i] = np.mean(hav_r[:,i])
    rstd_human[i] = np.std(hav_r[:,i])

    u2_human[i] = np.mean(hav_u2[:,i])
    u2std_human[i] = np.std(hav_u2[:,i])








#####################################################################################
#validation of pigeon model
#####################################################################################
Peps = pigeon_data['eps'].to_numpy()
Plr = pigeon_data['alpha'].to_numpy()
Puc = pigeon_data['lambda'].to_numpy()
Pgamr = pigeon_data['gamr'].to_numpy()

trials = pigeon_data['trials'].to_numpy()
choices = pigeon_data['choices'].to_numpy()

r_pigeon = np.zeros(len(Peps))
u2_pigeon = np.zeros(len(Peps))
rstd_pigeon = np.zeros(len(Peps))
u2std_pigeon = np.zeros(len(Peps))

pav_r = np.zeros((amount_of_sim,len(Peps)))
pav_u2 = np.zeros((amount_of_sim, len(Peps)))
for i in range(len(Peps)):
    eps = Peps[i]
    lr = Plr[i]
    uc = Puc[i]
    gamr = Pgamr[i]

    K = choices[i]
    buffer = K/2
    #make sure buffer is integer
    buffer = buffer.astype(int)
    T = trials[i]
    Q_int = 1

    for j in range(amount_of_sim):
        

        if K == 2:
            k, r, rand, Q_k_stored, delta_stored, Freq_all  = P2_RW_eps_lr_uc_gamr_adversarial(eps=eps, lr=lr, uc=uc, gamr=gamr, buffer=buffer, T=T, Q_int=Q_int)
            print('in loop of K = 2 and this is', i)

        if K == 4:
            k, r, rand, Q_k_stored, delta_stored, Freq_all = P4_RW_eps_lr_uc_gamr_adversarial(eps=eps, lr=lr, uc=uc, gamr=gamr, buffer=buffer, T=T, Q_int=Q_int)
            print('in loop of K = 4 and this is', i)
        if K == 8:
            k, r, rand, Q_k_stored, delta_stored, Freq_all = P8_RW_eps_lr_uc_gamr_adversarial(eps=eps, lr=lr, uc=uc, gamr=gamr, buffer=buffer, T=T, Q_int=Q_int)
            print('in loop of K = 8 and this is', i)

        pav_r[j,i] = np.mean(r)
        xo,pav_u2[j,i] = u_value(k[-500:], alternatives=K, two_alternatives=K*K)

    r_pigeon[i] = np.mean(pav_r[:,i])
    rstd_pigeon[i] = np.std(pav_r[:,i])

    u2_pigeon[i] = np.mean(pav_u2[:,i])
    u2std_pigeon[i] = np.std(pav_u2[:,i])



print('rewards are', r_pigeon)


#####################################################################################
#validation of rat model
#####################################################################################
Reps = rat_data['eps'].to_numpy()
Rlr = rat_data['alpha'].to_numpy()
Ruc = rat_data['lambda'].to_numpy()

trials = rat_data['trials'].to_numpy()
choices = rat_data['choices'].to_numpy()

r_rat = np.zeros(len(Reps))
u2_rat = np.zeros(len(Reps))
rstd_rat = np.zeros(len(Reps))
u2std_rat = np.zeros(len(Reps))
rav_r = np.zeros((amount_of_sim,len(Reps)))
rav_u2 = np.zeros((amount_of_sim, len(Reps)))

for i in range(len(Reps)):
    eps = Reps[i]
    lr = Rlr[i]
    uc = Ruc[i]

    K = choices[i]
    buffer = K/2
    #make sure buffer is integer
    buffer = buffer.astype(int)
    T = trials[i]
    Q_int = 1


    for j in range(amount_of_sim):

        if K == 5:
            k, r, rand, Q_k_stored, delta_stored  = R5_RW_eps_lr_uc_adversarial(eps=eps, lr=lr, uc=uc, T=T, Q_int=Q_int)
            print('in loop of K = 5 and this is', i)

        rav_r[j,i] = np.mean(r)
        xo,rav_u2[j,i] = u_value(k[-500:], alternatives=K, two_alternatives=K*K)



    r_rat[i] = np.mean(rav_r[:,i])
    rstd_rat[i] = np.std(rav_r[:,i])

    u2_rat[i] = np.mean(rav_u2[:,i])
    u2std_rat[i] = np.std(rav_u2[:,i])

#plot Q_k_stored
plt.figure(figsize=(10, 6))
for i in range(Q_k_stored.shape[1]):  # loop over 5 columns
    plt.plot(Q_k_stored[:, i], label=f'Option {i+1}')
plt.xlabel("trials")
plt.ylabel("Q-value")
plt.legend()
plt.grid(True)
plt.show()


################################################################################################
#real data human
################################################################################################
#Frist: Read Neuringer's data
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(script_dir)
data_dir = os.path.join(model_dir, '4Data_fit/ALL_DATA_CSV/Human')

#Read Neuringer's data
files = ['sub1con2.csv', 'sub2con2.csv', 'sub3con2.csv', 'sub4con2.csv', 'sub5con2.csv','sub6con2.csv', 'sub1con4.csv', 'sub2con4.csv', 'sub3con4.csv', 'sub4con4.csv', 'sub5con4.csv', 'sub6con4.csv', 'sub1con8.csv', 'sub2con8.csv', 'sub3con8.csv', 'sub4con8.csv', 'sub5con8.csv', 'sub6con8.csv']

r_human_real = np.zeros(len(files))
u2_human_real = np.zeros(len(files))
for sub, file in enumerate(files):
    sub_data = pd.read_csv(os.path.join(data_dir, file))
    r = sub_data['Reinforced'].to_numpy()
    #r = r[:500]
    k = sub_data['Choice'].to_numpy()
    k = k[-500:]
    alternatives = sub_data['Alternatives'].to_numpy()
    K = alternatives[3].astype(int)
    r_human_real[sub] = np.mean(r)
    u1_human_real, u2_human_real[sub] = u_value(k, alternatives=K, two_alternatives=K*K)



################################################################################################
#real data pigeon
################################################################################################
#Frist: Read Neuringer's data
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(script_dir)
data_dir = os.path.join(model_dir, '4Data_fit/ALL_DATA_CSV/Pigeon')
#Read Neuringer's data
files = ['sub1con2.csv', 'sub2con2.csv', 'sub3con2.csv', 'sub4con2.csv', 'sub5con2.csv', 'sub1con4.csv', 'sub2con4.csv', 'sub3con4.csv', 'sub4con4.csv', 'sub5con4.csv', 'sub1con8.csv', 'sub2con8.csv', 'sub3con8.csv', 'sub4con8.csv', 'sub5con8.csv']
r_pigeon_real = np.zeros(len(files))
u2_pigeon_real = np.zeros(len(files))
for sub, file in enumerate(files):
    sub_data = pd.read_csv(os.path.join(data_dir, file))
    r = sub_data['Reinforced'].to_numpy()
    #r = r[:500]
    k = sub_data['Choice'].to_numpy()
    k = k[-500:]
    alternatives = sub_data['Alternatives'].to_numpy()
    K = alternatives[3].astype(int)
    r_pigeon_real[sub] = np.mean(r)
    u1_pigeon_real, u2_pigeon_real[sub] = u_value(k, alternatives=K, two_alternatives=K*K)



################################################################################################
#real data rat
################################################################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(script_dir)
species_folder = os.path.join(model_dir, '4Data_fit/ALL_DATA_CSV/Rats')

subj_folders = os.listdir(species_folder)

r_rat_real = np.array([]) 
u2_rat_real = np.array([]) 
subj_array = np.array([])
for count, folder in enumerate(subj_folders):
    subj_dir = os.path.join(species_folder,f'{folder}')
    subj_data_list = os.listdir(subj_dir)
    for file_name in subj_data_list:
        data_dir = os.path.join(subj_dir, file_name)
        sub_data = pd.read_csv(data_dir)
        r = sub_data['choice'].to_numpy()
        #r = r[:500]
        k = sub_data['Resp'].to_numpy()
        k = k[-500:]
        K = 5
        r_rat_real = np.append(r_rat_real, np.mean(r))
        u1_rat_real, u2 = u_value(k, alternatives=K, two_alternatives=K*K)
        u2_rat_real = np.append(u2_rat_real, u2)
        subj_array = np.append(subj_array, folder)



#20 rats
av_r_rat = np.zeros(20)
av_u2_rat = np.zeros(20)
av_r_rat_real = np.zeros(20)
av_u2_rat_real = np.zeros(20)
subj_array = subj_array.astype(int)
for i in range(20):
    subj = i+1
    indices = np.where(subj_array==subj)[0]
    #av_r_rat[i] = np.mean(r_rat[indices])
    #av_u2_rat[i] = np.mean(u2_rat[indices])
    av_r_rat_real[i] = np.mean(r_rat_real[indices])
    av_u2_rat_real[i] = np.mean(u2_rat_real[indices])



matplotlib.rcParams['font.family'] = 'times new roman'
#bar plot
figsize=(7,2.5)
fontsize = 15
plt.figure(figsize=figsize)
labels = ['Human 2', 'Human 4', 'Human 8', 'Pigeon 2', 'Pigeon 4', 'Pigeon 8', 'Rat 5']
u2 = [np.mean(u2_human_real[:6]), np.mean(u2_human_real[6:12]), np.mean(u2_human_real[12:]), 
      np.mean(u2_pigeon_real[:5]), np.mean(u2_pigeon_real[5:10]), np.mean(u2_pigeon_real[10:]),
      np.mean(u2_rat_real)]
u2err = [np.std(u2_human_real[:6])/np.sqrt(6), np.std(u2_human_real[6:12])/np.sqrt(6), np.std(u2_human_real[12:])/np.sqrt(6), 
      np.std(u2_pigeon_real[:5])/np.sqrt(5), np.std(u2_pigeon_real[5:10])/np.sqrt(5), np.std(u2_pigeon_real[10:])/np.sqrt(5),
      np.std(u2_rat_real)/np.sqrt(len(u2_rat_real))]

colors = ["#493710", "#88661E", "#ddae48", "#08203B", "#285385", "#3f87da" , '#698286'] 
plt.bar(labels, u2, yerr= u2err, color=colors)
plt.ylabel('second-order entropy', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()







h2 = hav_r[:, :6].ravel()
h4 = hav_r[:, 6:12].ravel()
h8 = hav_r[:, 12:].ravel()

p2 = pav_r[:, :5].ravel()
p4 = pav_r[:, 5:10].ravel()
p8 = pav_r[:, 10:].ravel()

r5 = rav_r.ravel()

models = ("Human 2", "Human 4", "Human 8", "Pigeon 2", "Pigeon 4", "Pigeon 8", "Rat 5")
#HIER
species_means = {
    'Real': (np.mean(r_human_real[:6]), np.mean(r_human_real[6:12]), np.mean(r_human_real[12:]), 
      np.mean(r_pigeon_real[:5]), np.mean(r_pigeon_real[5:10]), np.mean(r_pigeon_real[10:]),
      np.mean(r_rat_real)),
    'Model prediction': (np.mean(h2), np.mean(h4), np.mean(h8), np.mean(p2), np.mean(p4), np.mean(p8), np.mean(r5)),
}



species_ste = {
    'Real': (np.std(r_human_real[:6])/np.sqrt(6), np.std(r_human_real[6:12])/np.sqrt(6), np.std(r_human_real[12:])/np.sqrt(6), 
      np.std(r_pigeon_real[:5])/np.sqrt(5), np.std(r_pigeon_real[5:10])/np.sqrt(5), np.std(r_pigeon_real[10:])/np.sqrt(5),
      np.std(r_rat_real)/np.sqrt(len(u2_rat_real))),
    'Model prediction': (np.std(h2)/np.sqrt(len(h2)), np.std(h4)/np.sqrt(len(h4)), np.std(h8)/np.sqrt(len(h8)), 
                         np.std(p2)/np.sqrt(len(p2)), np.std(p4)/np.sqrt(len(p4)), np.std(p8)/np.sqrt(len(p8)), 
                         np.std(r5)/np.sqrt(len(r5)))
}

x = np.arange(len(models))  # the label locations
width = 0.35 # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=figsize)

hatches = ["", "/"]
for multiplier, (attribute, measurement) in enumerate(species_means.items()):
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        measurement,
        width,
        label=attribute,
        color=colors[:len(models)],
        yerr=species_ste[attribute],
        capsize= 3,   # 7 category colors
        hatch=hatches[multiplier],
        edgecolor='black'   # pattern for Real vs Model
    )
    #ax.bar_label(rects, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('reward', fontsize=fontsize)
#ax.legend(loc='upper left', ncols=3)
ax.set_xticks(x + width / 2, models , fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1)




















h2 = hav_u2[:, :6].ravel()
h4 = hav_u2[:, 6:12].ravel()
h8 = hav_u2[:, 12:].ravel()

p2 = pav_u2[:, :5].ravel()
p4 = pav_u2[:, 5:10].ravel()
p8 = pav_u2[:, 10:].ravel()

r5 = rav_u2.ravel()

models = ("Human 2", "Human 4", "Human 8", "Pigeon 2", "Pigeon 4", "Pigeon 8", "Rat 5")
#HIER
species_means = {
    'Real': (np.mean(u2_human_real[:6]), np.mean(u2_human_real[6:12]), np.mean(u2_human_real[12:]), 
      np.mean(u2_pigeon_real[:5]), np.mean(u2_pigeon_real[5:10]), np.mean(u2_pigeon_real[10:]),
      np.mean(u2_rat_real)),
    'Model prediction': (np.mean(h2), np.mean(h4), np.mean(h8), np.mean(p2), np.mean(p4), np.mean(p8), np.mean(r5)),
}



species_ste = {
    'Real': (np.std(u2_human_real[:6])/np.sqrt(6), np.std(u2_human_real[6:12])/np.sqrt(6), np.std(u2_human_real[12:])/np.sqrt(6), 
      np.std(u2_pigeon_real[:5])/np.sqrt(5), np.std(u2_pigeon_real[5:10])/np.sqrt(5), np.std(u2_pigeon_real[10:])/np.sqrt(5),
      np.std(u2_rat_real)/np.sqrt(len(u2_rat_real))),
    'Model prediction': (np.std(h2)/np.sqrt(len(h2)), np.std(h4)/np.sqrt(len(h4)), np.std(h8)/np.sqrt(len(h8)), 
                         np.std(p2)/np.sqrt(len(p2)), np.std(p4)/np.sqrt(len(p4)), np.std(p8)/np.sqrt(len(p8)), 
                         np.std(r5)/np.sqrt(len(r5)))
}

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=figsize)

hatches = ["", "x"]
for multiplier, (attribute, measurement) in enumerate(species_means.items()):
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        measurement,
        width,
        label=attribute,
        color=colors[:len(models)],
        yerr=species_ste[attribute],
        capsize= 3,   # 7 category colors
        hatch=hatches[multiplier],
        alpha = 0.75,
        edgecolor = 'black'     # pattern for Real vs Model
    )
    #ax.bar_label(rects, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('second-order entropy', fontsize=fontsize)
#ax.legend(loc='upper left', ncols=3)
ax.set_xticks(x + width / 2, models , fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1)





models = ("Human 2", "Human 4", "Human 8", "Pigeon 2", "Pigeon 4", "Pigeon 8", "Rat 5")
#HIER
species_means = {
    'Real': (np.mean(u2_human_real[:6]), np.mean(u2_human_real[6:12]), np.mean(u2_human_real[12:]), 
      np.mean(u2_pigeon_real[:5]), np.mean(u2_pigeon_real[5:10]), np.mean(u2_pigeon_real[10:]),
      np.mean(u2_rat_real)),
    'Model prediction': (-1,-1,-1,-1,-1,-1,-1),
}



species_ste = {
    'Real': (np.std(u2_human_real[:6])/np.sqrt(6), np.std(u2_human_real[6:12])/np.sqrt(6), np.std(u2_human_real[12:])/np.sqrt(6), 
      np.std(u2_pigeon_real[:5])/np.sqrt(5), np.std(u2_pigeon_real[5:10])/np.sqrt(5), np.std(u2_pigeon_real[10:])/np.sqrt(5),
      np.std(u2_rat_real)/np.sqrt(len(u2_rat_real))),
    'Model prediction': (0,0,0,0,0,0,0)
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=figsize)

hatches = ["", "x"]
for multiplier, (attribute, measurement) in enumerate(species_means.items()):
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        measurement,
        width,
        label=attribute,
        color=colors[:len(models)], 
        yerr=species_ste[attribute],
        capsize= 3,   
        hatch=hatches[multiplier]     # pattern for Real vs Model
    )
    #ax.bar_label(rects, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('second-order entropy', fontsize=fontsize)
#ax.legend(loc='upper left', ncols=3)
ax.set_xticks(x + width / 40, models, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


'''
#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(r_human, r_human_real, c='#695a39', alpha=0.7)
plt.title(f'human reward')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()

print(r_human)
#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(u2_human, u2_human_real, c='#695a39', alpha=1)
plt.title(f'human second-order entropy')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()



#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(r_pigeon, r_pigeon_real, c="#3f87da", alpha=0.7)
plt.title(f'pigeon reward')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()

#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(u2_pigeon, u2_pigeon_real, c="#3f87da", alpha=1)
plt.title(f'pigeon second-order entropy')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()






#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(r_rat, r_rat_real, c="#698286", alpha=0.7)
plt.title(f'rat reward')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()

#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(u2_rat, u2_rat_real, c="#698286", alpha=1)
plt.title(f'rat second-order entropy')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()




#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(av_r_rat, av_r_rat_real, c="#698286", alpha=0.7)
plt.title(f'average rat reward')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()

#Scatter plots
plt.figure(figsize=(12, 8))
plt.scatter(av_u2_rat, av_u2_rat_real, c="#698286", alpha=1)
plt.title(f'average rat second-order entropy')
plt.xlabel('model prediction')
plt.ylabel('real')
plt.show()
















'''











































