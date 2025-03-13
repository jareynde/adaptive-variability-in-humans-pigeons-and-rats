#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Constant model
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
First reward schedule test:
    stable: [0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.3]
    volatile: [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1]
    adversarial: least frequent 60% of sequences
            
Critical parameters: epsilon, learning rate and unchosen addition.
This script will:
Calculate reward and entropy in each environment for all combinations of epsilon, learning rate and lambda (51 values per parameter).
Results are reported in heatmaps showing lr vs lambda, for each eps
@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random
from collections import Counter
import scipy.stats as stats
import itertools
import statsmodels.stats.multitest as smm


##############################################################################################################
#Functions of Rescorla-Wagner model, dfferent environments
##############################################################################################################
#simulation of Rescorla-Wagner model in an adversarial context
def RW_lambda_adversarial(Q_alpha, eps, unchosen, T, Q_int):
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #unchosen   --->        unchosen value-bias
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

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
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored


#simulation of Rescorla-Wagner model in stable context
def RW_lambda_stable(Q_alpha, eps, unchosen, T, Q_int, reward_stable):
    #alpha              --->        learning rate
    #eps                --->        epsilon
    #unchosen           --->        unchosen value-bias
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward_stable      --->        probabilites to recieve a reward
    
    K=8 #the amount of choice options    

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

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
        
        a1 = reward_stable[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored


#simulation of Rescorla-Wagner model in volatile context
def RW_lambda_volatile(Q_alpha, eps, unchosen, T, Q_int, reward_volatile):
    #alpha              --->        learning rate
    #eps                --->        epsilon
    #unchosen           --->        unchosen value-bias
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward_volatile    --->        probabilites to recieve a reward, shuffles among options

    K=8 #the amount of choice options
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000]     

    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
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
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, r, Q_k_stored





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


heat = 51

amount_of_sim = 10

T = 2500
Q_int = 1
reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]

all_eps = np.linspace(0, 1, heat, endpoint=True)
all_lr = np.linspace(0, 1, heat, endpoint=True)
all_unchosen = np.linspace(-1, 1, heat, endpoint=True)


#################################################################
#Simulations
#################################################################
done = 0
max_stable_rewards = np.zeros([heat, 4], dtype=float) #for every eps value there is one row, columns contain: max reward value, eps, learning rate, lambda
max_volatile_rewards = np.zeros([heat, 4], dtype=float) #for every eps value there is one row, columns contain: max reward value, eps, learning rate, lambda
max_adversarial_rewards = np.zeros([heat, 4], dtype=float) #for every eps value there is one row, columns contain: max reward value, eps, learning rate, lambda

for i in range(len(all_eps)): # vary eps
    #12 heatmaps for every epsilon value 
    mr_varheat = np.zeros([heat,heat]) #mean reward
    mr_volheat = np.zeros([heat,heat])
    mr_staheat = np.zeros([heat,heat])

    er_varheat = np.zeros([heat,heat]) #error reward
    er_volheat = np.zeros([heat,heat])
    er_staheat = np.zeros([heat,heat])

    mu_varheat = np.zeros([heat,heat]) #mean entropy
    mu_volheat = np.zeros([heat,heat])
    mu_staheat = np.zeros([heat,heat])

    eu_varheat = np.zeros([heat,heat]) #error entropy
    eu_volheat = np.zeros([heat,heat])
    eu_staheat = np.zeros([heat,heat])

    for j in range(heat): # vary learning rate
        for k in range(heat): #vary lambda
            done = done
            eps = all_eps[i]
            Q_alpha = all_lr[j]
            unchosen = all_unchosen[k]



            rsta = np.zeros(amount_of_sim, dtype = float)
            rvol = np.zeros(amount_of_sim, dtype=float)
            radv = np.zeros(amount_of_sim, dtype=float)

            usta = np.zeros(amount_of_sim, dtype = float)
            uvol = np.zeros(amount_of_sim, dtype=float)
            uadv = np.zeros(amount_of_sim, dtype=float)

            for sim in range(amount_of_sim):
                
                
                

                #in a stable environment
                c, r, Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable) #code: k stable (env) eps stable (optimal eps for stable)
                rsta[sim] = np.mean(r)
                u1, usta[sim] = u_value(seq=c[-500:], alternatives=8, two_alternatives=64)
                
                c, r, Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
                rvol[sim] = np.mean(r)
                u1, uvol[sim] = u_value(seq=c[-500:], alternatives=8, two_alternatives=64)

                c, r, Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int)
                radv[sim] = np.mean(r)
                u1, uadv[sim] = u_value(seq=c[-500:], alternatives=8, two_alternatives=64)

                

            
            mr_varheat[j,k] = np.mean(radv)
            mr_volheat[j,k] = np.mean(rvol)
            mr_staheat[j,k] = np.mean(rsta)

            er_varheat[j,k] = np.std(radv)
            er_volheat[j,k] = np.std(rvol)
            er_staheat[j,k] = np.std(rsta)

            mu_varheat[j,k] = np.mean(uadv)
            mu_volheat[j,k] = np.mean(uvol)
            mu_staheat[j,k] = np.mean(usta)

            eu_varheat[j,k] = np.std(uadv)
            eu_volheat[j,k] = np.std(uvol)
            eu_staheat[j,k] = np.std(usta)

            print(done)



    max_stable_rewards[i,0] = np.max(mr_staheat)
    max_indices = np.unravel_index(np.argmax(mr_staheat), mr_staheat.shape)
    max_stable_rewards[i,1] = eps
    max_stable_rewards[i,2] = all_lr[max_indices[0]]
    max_stable_rewards[i,3] = all_unchosen[max_indices[1]]


    max_volatile_rewards[i,0] = np.max(mr_volheat)
    max_indices = np.unravel_index(np.argmax(mr_volheat), mr_volheat.shape)
    max_volatile_rewards[i,1] = eps
    max_volatile_rewards[i,2] = all_lr[max_indices[0]]
    max_volatile_rewards[i,3] = all_unchosen[max_indices[1]]


    max_adversarial_rewards[i,0] = np.max(mr_varheat)
    max_indices = np.unravel_index(np.argmax(mr_varheat), mr_varheat.shape)
    max_adversarial_rewards[i,1] = eps
    max_adversarial_rewards[i,2] = all_lr[max_indices[0]]
    max_adversarial_rewards[i,3] = all_unchosen[max_indices[1]]
    #################################################################
    #Plotting + saving data
    #################################################################

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = "2Constant_heatmaps"
    save_dir = os.path.join(script_dir, folder_name)
    

    #reward heatmaps
    save_dir1 = os.path.join(save_dir, 'reward')

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(mr_varheat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel('lambda', fontsize=16)
    ax.set_ylabel('learning rate', fontsize=16)
    ax.set_xticks(ticks=list(range(heat)), labels=np.round(all_unchosen,1))  
    ax.set_yticks(ticks=list(range(heat)), labels=np.round(all_lr,1))  
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average reward')
    plt.title('adversarial environment')
    fig_name = os.path.join(save_dir1, f'adv_mr_2eps{i+43}.jpeg')
    plt.savefig(fig_name)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(mr_staheat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel('lambda', fontsize=16)
    ax.set_ylabel('learning rate', fontsize=16)
    ax.set_xticks(ticks=list(range(heat)), labels=np.round(all_unchosen,1))  
    ax.set_yticks(ticks=list(range(heat)), labels=np.round(all_lr,1))  
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average reward')  
    plt.title('stable environment')
    fig_name = os.path.join(save_dir1, f'sta_mr_2eps{i+43}.jpeg')
    plt.savefig(fig_name)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(mr_volheat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel('lambda', fontsize=16)
    ax.set_ylabel('learning rate', fontsize=16)
    ax.set_xticks(ticks=list(range(heat)), labels=np.round(all_unchosen,1))  
    ax.set_yticks(ticks=list(range(heat)), labels=np.round(all_lr,1))  
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average reward')  
    plt.title('volatile environment')
    fig_name = os.path.join(save_dir1, f'vol_mr_2eps{i+43}.jpeg')
    plt.savefig(fig_name)

    #save reward data per epsilon
    df_mr_varheat = pd.DataFrame(mr_varheat)
    df_mr_varheat.to_excel(excel_writer=(os.path.join(save_dir1, f'adv_mr_eps{i}.xlsx')))

    df_mr_staheat = pd.DataFrame(mr_staheat)
    df_mr_staheat.to_excel(excel_writer=(os.path.join(save_dir1, f'sta_mr_eps{i}.xlsx')))

    df_mr_volheat = pd.DataFrame(mr_volheat)
    df_mr_volheat.to_excel(excel_writer=(os.path.join(save_dir1, f'vol_mr_eps{i}.xlsx')))



    df_er_varheat = pd.DataFrame(er_varheat)
    df_er_varheat.to_excel(excel_writer=(os.path.join(save_dir1, f'adv_er_eps{i}.xlsx')))

    df_er_staheat = pd.DataFrame(er_staheat)
    df_er_staheat.to_excel(excel_writer=(os.path.join(save_dir1, f'sta_er_eps{i}.xlsx')))

    df_er_volheat = pd.DataFrame(er_volheat)
    df_er_volheat.to_excel(excel_writer=(os.path.join(save_dir1, f'vol_er_eps{i}.xlsx')))







    #entropy heatmaps
    save_dir2 = os.path.join(save_dir, 'entropy')

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(mu_varheat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel('lambda', fontsize=16)
    ax.set_ylabel('learning rate', fontsize=16)
    ax.set_xticks(ticks=list(range(heat)), labels=np.round(all_unchosen,1))  
    ax.set_yticks(ticks=list(range(heat)), labels=np.round(all_lr,1))  
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average entropy')
    plt.title('adversarial environment')
    fig_name = os.path.join(save_dir2, f'adv_mr_2eps{i+43}.jpeg')
    plt.savefig(fig_name)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(mu_staheat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel('lambda', fontsize=16)
    ax.set_ylabel('learning rate', fontsize=16)
    ax.set_xticks(ticks=list(range(heat)), labels=np.round(all_unchosen,1))  
    ax.set_yticks(ticks=list(range(heat)), labels=np.round(all_lr,1))  
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average entropy')  
    plt.title('stable environment')
    fig_name = os.path.join(save_dir2, f'sta_mr_2eps{i+43}.jpeg')
    plt.savefig(fig_name)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(mu_volheat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xlabel('lambda', fontsize=16)
    ax.set_ylabel('learning rate', fontsize=16)
    ax.set_xticks(ticks=list(range(heat)), labels=np.round(all_unchosen,1))  
    ax.set_yticks(ticks=list(range(heat)), labels=np.round(all_lr,1))  
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average entropy')  
    plt.title('volatile environment')
    fig_name = os.path.join(save_dir2, f'vol_mr_2eps{i+43}.jpeg')
    plt.savefig(fig_name)

    #save reward data per epsilon
    df_mu_varheat = pd.DataFrame(mu_varheat)
    df_mu_varheat.to_excel(os.path.join(save_dir2, f'adv_mu_2eps{i+43}.xlsx'))

    df_mu_staheat = pd.DataFrame(mu_staheat)
    df_mu_staheat.to_excel(os.path.join(save_dir2, f'sta_mu_2eps{i+43}.xlsx'))

    df_mu_volheat = pd.DataFrame(mu_volheat)
    df_mu_volheat.to_excel(os.path.join(save_dir2, f'vol_mu_2eps{i+43}.xlsx'))



    df_eu_varheat = pd.DataFrame(eu_varheat)
    df_eu_varheat.to_excel(os.path.join(save_dir2, f'adv_eu_2eps{i+43}.xlsx'))

    df_eu_staheat = pd.DataFrame(eu_staheat)
    df_eu_staheat.to_excel(os.path.join(save_dir2, f'sta_eu_2eps{i+43}.xlsx'))

    df_eu_volheat = pd.DataFrame(eu_volheat)
    df_eu_volheat.to_excel(os.path.join(save_dir2, f'vol_eu_2eps{i+43}.xlsx'))











df_max_stable_rewards = pd.DataFrame(max_stable_rewards)
df_max_stable_rewards.to_excel(os.path.join(save_dir, 'max_stable_rewards.xlsx'))

df_max_volatile_rewards = pd.DataFrame(max_volatile_rewards)
df_max_volatile_rewards.to_excel(os.path.join(save_dir, 'max_volatile_rewards.xlsx'))

df_max_adversarial_rewards = pd.DataFrame(max_adversarial_rewards)
df_max_adversarial_rewards.to_excel(os.path.join(save_dir, 'max_adversarial_rewards.xlsx'))
