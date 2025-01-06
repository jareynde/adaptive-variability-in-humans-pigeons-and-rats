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
*Plot the U-values for the generated sequences in each context
*Compare performance (reward and u-value) of a model with a lambda that is constant across the three environments (three constant model simulations with optimal LR value for stable, volatile and adversarial)

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



#Optimal lambda's in each environment
#From the ML_eps script: average of the last 100 epsilons for all ML simulations
ld_sta = -0.21
ld_vol = 0.24 
ld_adv = 0.83

#0.834020193	-0.207531197	0.239501203



Q_alpha = 0.25
eps = 0.3


T = 2500
amount_of_sim = 100

reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
Q_int = 1

seeds = list(range(3*amount_of_sim))  # Random seeds


#################################################################
#Simulations
#################################################################



kslds = np.zeros([amount_of_sim, T])
rslds = np.zeros([amount_of_sim, T])
ksldv = np.zeros([amount_of_sim, T])
rsldv = np.zeros([amount_of_sim, T])
kslda = np.zeros([amount_of_sim, T])
rslda = np.zeros([amount_of_sim, T])

kvlds = np.zeros([amount_of_sim, T])
rvlds = np.zeros([amount_of_sim, T])
kvldv = np.zeros([amount_of_sim, T])
rvldv = np.zeros([amount_of_sim, T])
kvlda = np.zeros([amount_of_sim, T])
rvlda = np.zeros([amount_of_sim, T])

kalds = np.zeros([amount_of_sim, T])
ralds = np.zeros([amount_of_sim, T])
kaldv = np.zeros([amount_of_sim, T])
raldv = np.zeros([amount_of_sim, T])
kalda = np.zeros([amount_of_sim, T])
ralda = np.zeros([amount_of_sim, T])


u1slds = np.zeros(amount_of_sim)
u2slds = np.zeros(amount_of_sim)
u1sldv = np.zeros(amount_of_sim)
u2sldv = np.zeros(amount_of_sim)
u1slda = np.zeros(amount_of_sim)
u2slda = np.zeros(amount_of_sim)

u1vlds = np.zeros(amount_of_sim)
u2vlds = np.zeros(amount_of_sim)
u1vldv = np.zeros(amount_of_sim)
u2vldv = np.zeros(amount_of_sim)
u1vlda = np.zeros(amount_of_sim)
u2vlda = np.zeros(amount_of_sim)

u1alds = np.zeros(amount_of_sim)
u2alds = np.zeros(amount_of_sim)
u1aldv = np.zeros(amount_of_sim)
u2aldv = np.zeros(amount_of_sim)
u1alda = np.zeros(amount_of_sim)
u2alda = np.zeros(amount_of_sim)


for sim in range(amount_of_sim):

    random.seed(seeds[sim])
    np.random.seed(seeds[sim])

    #in a stable environment
    kslds[sim,:], rslds[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps, unchosen=ld_sta, T=T, Q_int=Q_int, reward_stable=reward_stable) #code: k stable (env) eps stable (optimal eps for stable)
    u1slds[sim], u2slds[sim] = u_value(seq=kslds[sim,:], alternatives=8, two_alternatives=64)

    ksldv[sim,:], rsldv[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps, unchosen=ld_vol, T=T, Q_int=Q_int, reward_stable=reward_stable)
    u1sldv[sim], u2sldv[sim] = u_value(seq=ksldv[sim,:], alternatives=8, two_alternatives=64)

    kslda[sim,:], rslda[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps, unchosen=ld_adv, T=T, Q_int=Q_int, reward_stable=reward_stable)
    u1slda[sim], u2slda[sim] = u_value(seq=kslda[sim,:], alternatives=8, two_alternatives=64)

    random.seed(seeds[sim+amount_of_sim])
    np.random.seed(seeds[sim+amount_of_sim])

    #in a volatile environment
    kvlds[sim,:], rvlds[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps, unchosen=ld_sta, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vlds[sim], u2vlds[sim] = u_value(seq=kvlds[sim,:], alternatives=8, two_alternatives=64)

    kvldv[sim,:], rvldv[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps, unchosen=ld_vol, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vldv[sim], u2vldv[sim] = u_value(seq=kvldv[sim,:], alternatives=8, two_alternatives=64)

    kvlda[sim,:], rvlda[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps, unchosen=ld_adv, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vlda[sim], u2vlda[sim] = u_value(seq=kvlda[sim,:], alternatives=8, two_alternatives=64)

    random.seed(seeds[sim+2*amount_of_sim])
    np.random.seed(seeds[sim+2*amount_of_sim])

    #in an adversarial environment
    kalds[sim,:], ralds[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps, unchosen=ld_sta, T=T, Q_int=Q_int)
    u1alds[sim], u2alds[sim] = u_value(seq=kalds[sim,:], alternatives=8, two_alternatives=64)

    kaldv[sim,:], raldv[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps, unchosen=ld_vol, T=T, Q_int=Q_int)
    u1aldv[sim], u2aldv[sim] = u_value(seq=kaldv[sim,:], alternatives=8, two_alternatives=64)

    kalda[sim,:], ralda[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps, unchosen=ld_adv, T=T, Q_int=Q_int)
    u1alda[sim], u2alda[sim] = u_value(seq=kalda[sim,:], alternatives=8, two_alternatives=64)

    print('check',sim)


#Stable
sim_mean_rslds = np.mean(rslds, axis=1)
mean_rslds = np.mean(sim_mean_rslds)
std_rslds = np.std(sim_mean_rslds)
ste_rslds = std_rslds/np.sqrt(amount_of_sim)

mean_u2slds = np.mean(u2slds)
std_u2slds = np.std(u2slds)
ste_u2slds = std_u2slds/np.sqrt(amount_of_sim)


sim_mean_rsldv = np.mean(rsldv, axis=1)
mean_rsldv = np.mean(sim_mean_rsldv)
std_rsldv = np.std(sim_mean_rsldv)
ste_rsldv = std_rsldv/np.sqrt(amount_of_sim)

mean_u2sldv = np.mean(u2sldv)
std_u2sldv = np.std(u2sldv)
ste_u2sldv = std_u2sldv/np.sqrt(amount_of_sim)


sim_mean_rslda = np.mean(rslda, axis=1)
mean_rslda = np.mean(sim_mean_rslda)
std_rslda = np.std(sim_mean_rslda)
ste_rslda = std_rslda/np.sqrt(amount_of_sim)

mean_u2slda = np.mean(u2slda)
std_u2slda = np.std(u2slda)
ste_u2slda = std_u2slda/np.sqrt(amount_of_sim)




#Volatile
sim_mean_rvlds = np.mean(rvlds, axis=1)
mean_rvlds = np.mean(sim_mean_rvlds)
std_rvlds = np.std(sim_mean_rvlds)
ste_rvlds = std_rvlds/np.sqrt(amount_of_sim)

mean_u2vlds = np.mean(u2vlds)
std_u2vlds = np.std(u2vlds)
ste_u2vlds = std_u2vlds/np.sqrt(amount_of_sim)


sim_mean_rvldv = np.mean(rvldv, axis=1)
mean_rvldv = np.mean(sim_mean_rvldv)
std_rvldv = np.std(sim_mean_rvldv)
ste_rvldv = std_rvldv/np.sqrt(amount_of_sim)

mean_u2vldv = np.mean(u2vldv)
std_u2vldv = np.std(u2vldv)
ste_u2vldv = std_u2vldv/np.sqrt(amount_of_sim)


sim_mean_rvlda = np.mean(rvlda, axis=1)
mean_rvlda = np.mean(sim_mean_rvlda)
std_rvlda = np.std(sim_mean_rvlda)
ste_rvlda = std_rvlda/np.sqrt(amount_of_sim)

mean_u2vlda = np.mean(u2vlda)
std_u2vlda = np.std(u2vlda)
ste_u2vlda = std_u2vlda/np.sqrt(amount_of_sim)





#adversarial
sim_mean_ralds = np.mean(ralds, axis=1)
mean_ralds = np.mean(sim_mean_ralds)
std_ralds = np.std(sim_mean_ralds)
ste_ralds = std_ralds/np.sqrt(amount_of_sim)

mean_u2alds = np.mean(u2alds)
std_u2alds = np.std(u2alds)
ste_u2alds = std_u2alds/np.sqrt(amount_of_sim)


sim_mean_raldv = np.mean(raldv, axis=1)
mean_raldv = np.mean(sim_mean_raldv)
std_raldv = np.std(sim_mean_raldv)
ste_raldv = std_raldv/np.sqrt(amount_of_sim)

mean_u2aldv = np.mean(u2aldv)
std_u2aldv = np.std(u2aldv)
ste_u2aldv = std_u2aldv/np.sqrt(amount_of_sim)


sim_mean_ralda = np.mean(ralda, axis=1)
mean_ralda = np.mean(sim_mean_ralda)
std_ralda = np.std(sim_mean_ralda)
ste_ralda = std_ralda/np.sqrt(amount_of_sim)

mean_u2alda = np.mean(u2alda)
std_u2alda = np.std(u2alda)
ste_u2alda = std_u2alda/np.sqrt(amount_of_sim)


###########################################################################################################################
#PLOTTING
###########################################################################################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = "2Constant_lambda"
save_dir = os.path.join(script_dir, folder_name)

figsize = (5,7)
random.seed(5454654865)
np.random.seed(569698)

#Figure 1: REWARDS GROUPED PER EPSILON VALUE
#plot rewards for each situation
#first bars are grouped per epsilon value
groups = [fr'$\lambda$ = {ld_sta} (optimal for stable)', fr'$\lambda$ = {ld_vol} (optimal for volatile)', fr'$\lambda$ = {ld_adv} (optimal for adversarial)']
categories = ['stable environment', 'volatile environment', 'adversarial environment']
values = np.array([[mean_rslds, mean_rvlds, mean_ralds],
                   [mean_rsldv, mean_rvldv, mean_raldv],
                   [mean_rslda, mean_rvlda, mean_ralda]])
errors = np.array([[ste_rslds, ste_rvlds, ste_ralds],
                   [ste_rsldv, ste_rvldv, ste_raldv],
                   [ste_rslda, ste_rvlda, ste_ralda]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    # Plot the bars for the category
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], edgecolor = 'gold', linewidth=1.3, alpha=0.5, label=f'{category}' if i == 0 else "")

# Customize the plot
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Reward')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
#ax.legend()
fig_name = os.path.join(save_dir, f'reward_per_lambda')
plt.savefig(fig_name)
plt.show()


#Figure 2: REWARDS GROUPED PER ENVIRONMENT
#second bars are grouped per environment
categories = [fr'$\lambda$ = {ld_sta} (optimal for stable)', fr'$\lambda$ = {ld_vol} (optimal for volatile)', fr'$\lambda$ = {ld_adv} (optimal for adversarial)']
groups = ['stable environment', 'volatile environment', 'adversarial environment']

values = np.array([[mean_rslds,mean_rsldv,mean_rslda],
                  [mean_rvlds,mean_rvldv,mean_rvlda],
                  [mean_ralds,mean_raldv,mean_ralda]])

errors = np.array([[ste_rslds,ste_rsldv,ste_rslda],
                  [ste_rvlds,ste_rvldv,ste_rvlda],
                  [ste_ralds,ste_raldv,ste_ralda]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    # Plot the bars for the category
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], edgecolor = 'gold', linewidth=1.3,alpha=0.5, label=f'{category}' if i == 0 else "")

# Customize the plot
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Reward')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
#ax.legend()
fig_name = os.path.join(save_dir, f'reward_per_env')
plt.savefig(fig_name)
plt.show()



#Figure 3: OPTIMAL U-VALUES
#plot u-values
#optimal u-values
category_colors = ['darkcyan', 'darkorange', 'forestgreen']
categories = ['stable', 'volatile', 'adversarial']
values = [mean_u2slds, mean_u2vldv, mean_u2alda]
errors = [ste_u2slds, ste_u2vldv, ste_u2alda]
fig, ax = plt.subplots(figsize=figsize)
ax.bar(categories, values, color=category_colors, edgecolor='gold', alpha=0.75, yerr=errors, capsize=5)
ax.set_xlabel('environment')
ax.set_ylabel('U-Values')
ax.set_ylim([0,1])
fig_name = os.path.join(save_dir, f'u_values_opt')
plt.savefig(fig_name)
plt.show()

#Figure 4: ALL U-VALUES
#all u-values
groups = [r'optimal for stable', r'optimal for volatile', r'optimal for adversarial']
categories = ['stable environment', 'volatile environment', 'adversarial environment']
values = np.array([[mean_u2slds, mean_u2vlds, mean_u2alds],
                   [mean_u2sldv, mean_u2vldv, mean_u2aldv],
                   [mean_u2slda, mean_u2vlda, mean_u2alda]])

errors = np.array([[ste_u2slds, ste_u2vlds, ste_u2alds],
                   [ste_u2sldv, ste_u2vldv, ste_u2aldv],
                   [ste_u2slda, ste_u2vlda, ste_u2alda]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    # Plot the bars for the category
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], alpha=0.75, label=f'{category}' if i == 0 else "")

# Customize the plot
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('U-values')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
ax.legend()
fig_name = os.path.join(save_dir, f'u_values_all')
plt.savefig(fig_name)
plt.show()



#Figure 5: AVERAGE REWARDS ACROSS ENVIRONMENTS
#plot rewards as an average across three environments
R_lds = np.concatenate((rslds, rvlds, ralds), axis=0)
R_ldv = np.concatenate((rsldv, rvldv, raldv), axis=0)
R_lda = np.concatenate((rslda, rvlda, ralda), axis=0)
RML = np.concatenate((rslds, rvldv, ralda), axis=0)

sim_mean_R_lds = np.mean(R_lds, axis=1)
mean_R_lds = np.mean(sim_mean_R_lds)
std_R_lds = np.std(sim_mean_R_lds)
ste_R_lds = std_R_lds/np.sqrt(3*amount_of_sim)

sim_mean_R_ldv = np.mean(R_ldv, axis=1)
mean_R_ldv = np.mean(sim_mean_R_ldv)
std_R_ldv = np.std(sim_mean_R_ldv)
ste_R_ldv = std_R_ldv/np.sqrt(3*amount_of_sim)

sim_mean_R_lda = np.mean(R_lda, axis=1)
mean_R_lda = np.mean(sim_mean_R_lda)
std_R_lda = np.std(sim_mean_R_lda)
ste_R_lda = std_R_lda/np.sqrt(3*amount_of_sim)

sim_mean_RML = np.mean(RML, axis=1)
mean_RML = np.mean(sim_mean_RML)
std_RML = np.std(sim_mean_RML)
ste_RML = std_RML/np.sqrt(3*amount_of_sim)



categories = [r'stable-optimal $\lambda$', r'volatile-optimal $\lambda$', r'adversarial-optimal $\lambda$', r'environment-specific $\lambda$']
values = [mean_R_lds, mean_R_ldv, mean_R_lda, mean_RML]
errors = [ste_R_lds, ste_R_ldv, ste_R_lda, ste_RML]

# Create the bar plot with error bars
fig, ax = plt.subplots(figsize=figsize)
ax.bar(categories, values, color='gold', edgecolor='black', alpha=0.5, linewidth=1.2, yerr=errors, capsize=5)

# Customize the plot
ax.set_xlabel('environment')
ax.set_ylabel('Reward')
ax.set_ylim([0,1])
fig_name = os.path.join(save_dir, f'reward_average')
plt.savefig(fig_name)
plt.show()






###########################################################################################################################
#STATS
###########################################################################################################################
#these are comparisons for not normally distributed populations with unequal means

#do statistical tests:
#Shapiro to test normality
#Levene to test equal variances
#Mann-Whitney U Test when data is not normally distributed
#Welch’s t-test when data is normally distributed but unequal variances
#Independent t-test when data is normally distributed and equal variances
#######################################
#Difference between average rewards across environments?
#######################################
print('OPT LAMBDA DIFFERENCES IN REWARDS')
#check normality:
_, p1 = stats.shapiro(sim_mean_R_lds)
_, p2 = stats.shapiro(sim_mean_R_ldv)
_, p3 = stats.shapiro(sim_mean_R_lda)
_, p4 = stats.shapiro(sim_mean_RML)


#check for equal variances:
_, p = stats.levene(sim_mean_R_lds, sim_mean_R_ldv, sim_mean_R_lda, sim_mean_RML)

if p1 < 0.05 or p2 < 0.05 or p3 < 0.05 or p4 < 0.05:
    normality = False
else: normality = True


if p < 0.05 :
    equal_variances = False
else: equal_variances = True

print(f"P-values for normality: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}, Group 4: {p4}, and so normality is {normality}")
print(f"P-value for equal variances: {p}, and so equal_variances are {equal_variances}")


# Assuming data is in arrays group1 (reference group), group2, group3, and group4
reference_group = sim_mean_RML
other_groups = [sim_mean_R_lds, sim_mean_R_ldv, sim_mean_R_lda]
group_names = ['stable', 'volatile', 'adversarial']

p_values = []

# Perform Mann-Whitney U test for each comparison between the reference group and each other group
for other_group, group_name in zip(other_groups, group_names):
    if normality == False:
        _, p = stats.mannwhitneyu(reference_group, other_group, alternative='two-sided')
        #print(f"P-value for Reference Group vs {group_name}: {p}")
        p_values.append(p)
        print('we did the normality is false if')

    elif (normality == True) and (equal_variances == False):
            _, p = stats.ttest_ind(reference_group, other_group, equal_var=False)  # Welch's t-test
            #print(f"P-value for {n1} vs {n2}: {p}")
            p_values.append(p)
            print('we did the normality is true and equal variances is false if')


    elif (normality == True) and (equal_variances == True):
        _, p = stats.ttest_ind(reference_group, other_group, equal_var=True)  
        #print(f"P-value for {n1} vs {n2}: {p}")
        p_values.append(p)
        print('we did the normality is true and equal variances is true if')

# Correct for multiple comparisons
reject, pvals_corrected, _, _ = smm.multipletests(p_values, method='bonferroni')
#print(f"Corrected P-values: {pvals_corrected}")

# Interpret the results
for group_name, p_corr, rej in zip(group_names, pvals_corrected, reject):
    result = "significant" if rej else "not significant"
    print(f"Comparison env-specific lambda vs {group_name} is {result} (Corrected P-value: {p_corr})")


#######################################
#Difference between average U-values across environments?
#######################################
print('OPT LAMBDA DIFFERENCES IN U-VALUES')
#check normality:
_, p1 = stats.shapiro(u2slds)
_, p2 = stats.shapiro(u2vldv)
_, p3 = stats.shapiro(u2alda)

#check for equal variances:
_, p = stats.levene(u2slds, u2vldv, u2alda)

if p1 < 0.05 or p2 < 0.05 or p3 < 0.05:
    normality = False
else: normality = True

if p < 0.05 :
    equal_variances = False
else: equal_variances = True

print(f"P-values for normality: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}, and so normality is {normality}")
print(f"P-value for equal variances: {p}, and so equal_variance is {equal_variances}")


# Assuming data is in arrays group1, group2, and group3
groups = [u2slds, u2vldv, u2alda]
group_names = ['stable U', 'volatile U', 'adversarial U']
pairwise_comparisons = list(itertools.combinations(groups, 2))
pairwise_names = list(itertools.combinations(group_names, 2))

p_values = []

# Perform Mann-Whitney U test for each pair of groups
for (g1, g2), (n1, n2) in zip(pairwise_comparisons, pairwise_names):
    if normality == False:
        _, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        #print(f"P-value for {n1} vs {n2}: {p}")
        p_values.append(p)
        print('we did the normality is false if')

    elif (normality == True) and (equal_variances == False):
         _, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
         #print(f"P-value for {n1} vs {n2}: {p}")
         p_values.append(p)
         print('we did the normality is true and equal variances is false if')

    elif (normality == True) and (equal_variances == True):
        _, p = stats.ttest_ind(g1, g2, equal_var=True) 
         #print(f"P-value for {n1} vs {n2}: {p}")
        p_values.append(p)
        print('we did the normality is true and equal variances is true if')

         
# Correct for multiple comparisons
reject, pvals_corrected, _, _ = smm.multipletests(p_values, method='bonferroni')
#print(f"Corrected P-values: {pvals_corrected}")

# Interpret the results
for (n1, n2), p_corr, rej in zip(pairwise_names, pvals_corrected, reject):
    result = "significant" if rej else "not significant"
    print(f"Comparison {n1} vs {n2} is {result} (Corrected P-value: {p_corr})")


