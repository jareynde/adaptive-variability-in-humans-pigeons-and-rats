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
import matplotlib    
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




#Optimal learning rates in each environment
#From the ML_eps script: average of the last 100 epsilons for all ML simulations
lr_sta = 0.31
lr_vol = 0.87
lr_adv = 0.98






eps = 0.24
unchosen = 0

T = 2500
amount_of_sim = 100

reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
Q_int = 1

seeds = list(range(3*amount_of_sim))  # Random seeds


#################################################################
#Simulations
#################################################################



kslrs = np.zeros([amount_of_sim, T])
rslrs = np.zeros([amount_of_sim, T])
kslrv = np.zeros([amount_of_sim, T])
rslrv = np.zeros([amount_of_sim, T])
kslra = np.zeros([amount_of_sim, T])
rslra = np.zeros([amount_of_sim, T])

kvlrs = np.zeros([amount_of_sim, T])
rvlrs = np.zeros([amount_of_sim, T])
kvlrv = np.zeros([amount_of_sim, T])
rvlrv = np.zeros([amount_of_sim, T])
kvlra = np.zeros([amount_of_sim, T])
rvlra = np.zeros([amount_of_sim, T])

kalrs = np.zeros([amount_of_sim, T])
ralrs = np.zeros([amount_of_sim, T])
kalrv = np.zeros([amount_of_sim, T])
ralrv = np.zeros([amount_of_sim, T])
kalra = np.zeros([amount_of_sim, T])
ralra = np.zeros([amount_of_sim, T])


u1slrs = np.zeros(amount_of_sim)
u2slrs = np.zeros(amount_of_sim)
u1slrv = np.zeros(amount_of_sim)
u2slrv = np.zeros(amount_of_sim)
u1slra = np.zeros(amount_of_sim)
u2slra = np.zeros(amount_of_sim)

u1vlrs = np.zeros(amount_of_sim)
u2vlrs = np.zeros(amount_of_sim)
u1vlrv = np.zeros(amount_of_sim)
u2vlrv = np.zeros(amount_of_sim)
u1vlra = np.zeros(amount_of_sim)
u2vlra = np.zeros(amount_of_sim)

u1alrs = np.zeros(amount_of_sim)
u2alrs = np.zeros(amount_of_sim)
u1alrv = np.zeros(amount_of_sim)
u2alrv = np.zeros(amount_of_sim)
u1alra = np.zeros(amount_of_sim)
u2alra = np.zeros(amount_of_sim)



for sim in range(amount_of_sim):

    random.seed(seeds[sim])
    np.random.seed(seeds[sim])


    #in a stable environment
    kslrs[sim,:], rslrs[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=lr_sta, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable) #code: k stable (env) eps stable (optimal eps for stable)
    u1slrs[sim], u2slrs[sim] = u_value(seq=kslrs[sim,:], alternatives=8, two_alternatives=64)

    kslrv[sim,:], rslrv[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=lr_vol, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable)
    u1slrv[sim], u2slrv[sim] = u_value(seq=kslrv[sim,:], alternatives=8, two_alternatives=64)

    kslra[sim,:], rslra[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=lr_adv, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable)
    u1slra[sim], u2slra[sim] = u_value(seq=kslra[sim,:], alternatives=8, two_alternatives=64)

    random.seed(seeds[sim+amount_of_sim])
    np.random.seed(seeds[sim+amount_of_sim])


    #in a volatile environment
    kvlrs[sim,:], rvlrs[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=lr_sta, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vlrs[sim], u2vlrs[sim] = u_value(seq=kvlrs[sim,:], alternatives=8, two_alternatives=64)

    kvlrv[sim,:], rvlrv[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=lr_vol, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vlrv[sim], u2vlrv[sim] = u_value(seq=kvlrv[sim,:], alternatives=8, two_alternatives=64)

    kvlra[sim,:], rvlra[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=lr_adv, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vlra[sim], u2vlra[sim] = u_value(seq=kvlra[sim,:], alternatives=8, two_alternatives=64)

    random.seed(seeds[sim+2*amount_of_sim])
    np.random.seed(seeds[sim+2*amount_of_sim])

    #in an adversarial environment
    kalrs[sim,:], ralrs[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=lr_sta, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int)
    u1alrs[sim], u2alrs[sim] = u_value(seq=kalrs[sim,:], alternatives=8, two_alternatives=64)

    kalrv[sim,:], ralrv[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=lr_vol, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int)
    u1alrv[sim], u2alrv[sim] = u_value(seq=kalrv[sim,:], alternatives=8, two_alternatives=64)

    kalra[sim,:], ralra[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=lr_adv, eps=eps, unchosen=unchosen, T=T, Q_int=Q_int)
    u1alra[sim], u2alra[sim] = u_value(seq=kalra[sim,:], alternatives=8, two_alternatives=64)

    print('check',sim)


#Stable
sim_mean_rslrs = np.mean(rslrs, axis=1)
mean_rslrs = np.mean(sim_mean_rslrs)
std_rslrs = np.std(sim_mean_rslrs)
ste_rslrs = std_rslrs/np.sqrt(amount_of_sim)

mean_u2slrs = np.mean(u2slrs)
std_u2slrs = np.std(u2slrs)
ste_u2slrs = std_u2slrs/np.sqrt(amount_of_sim)


sim_mean_rslrv = np.mean(rslrv, axis=1)
mean_rslrv = np.mean(sim_mean_rslrv)
std_rslrv = np.std(sim_mean_rslrv)
ste_rslrv = std_rslrv/np.sqrt(amount_of_sim)

mean_u2slrv = np.mean(u2slrv)
std_u2slrv = np.std(u2slrv)
ste_u2slrv = std_u2slrv/np.sqrt(amount_of_sim)


sim_mean_rslra = np.mean(rslra, axis=1)
mean_rslra = np.mean(sim_mean_rslra)
std_rslra = np.std(sim_mean_rslra)
ste_rslra = std_rslra/np.sqrt(amount_of_sim)

mean_u2slra = np.mean(u2slra)
std_u2slra = np.std(u2slra)
ste_u2slra = std_u2slra/np.sqrt(amount_of_sim)




#Volatile
sim_mean_rvlrs = np.mean(rvlrs, axis=1)
mean_rvlrs = np.mean(sim_mean_rvlrs)
std_rvlrs = np.std(sim_mean_rvlrs)
ste_rvlrs = std_rvlrs/np.sqrt(amount_of_sim)

mean_u2vlrs = np.mean(u2vlrs)
std_u2vlrs = np.std(u2vlrs)
ste_u2vlrs = std_u2vlrs/np.sqrt(amount_of_sim)


sim_mean_rvlrv = np.mean(rvlrv, axis=1)
mean_rvlrv = np.mean(sim_mean_rvlrv)
std_rvlrv = np.std(sim_mean_rvlrv)
ste_rvlrv = std_rvlrv/np.sqrt(amount_of_sim)

mean_u2vlrv = np.mean(u2vlrv)
std_u2vlrv = np.std(u2vlrv)
ste_u2vlrv = std_u2vlrv/np.sqrt(amount_of_sim)


sim_mean_rvlra = np.mean(rvlra, axis=1)
mean_rvlra = np.mean(sim_mean_rvlra)
std_rvlra = np.std(sim_mean_rvlra)
ste_rvlra = std_rvlra/np.sqrt(amount_of_sim)

mean_u2vlra = np.mean(u2vlra)
std_u2vlra = np.std(u2vlra)
ste_u2vlra = std_u2vlra/np.sqrt(amount_of_sim)





#adversarial
sim_mean_ralrs = np.mean(ralrs, axis=1)
mean_ralrs = np.mean(sim_mean_ralrs)
std_ralrs = np.std(sim_mean_ralrs)
ste_ralrs = std_ralrs/np.sqrt(amount_of_sim)

mean_u2alrs = np.mean(u2alrs)
std_u2alrs = np.std(u2alrs)
ste_u2alrs = std_u2alrs/np.sqrt(amount_of_sim)


sim_mean_ralrv = np.mean(ralrv, axis=1)
mean_ralrv = np.mean(sim_mean_ralrv)
std_ralrv = np.std(sim_mean_ralrv)
ste_ralrv = std_ralrv/np.sqrt(amount_of_sim)

mean_u2alrv = np.mean(u2alrv)
std_u2alrv = np.std(u2alrv)
ste_u2alrv = std_u2alrv/np.sqrt(amount_of_sim)


sim_mean_ralra = np.mean(ralra, axis=1)
mean_ralra = np.mean(sim_mean_ralra)
std_ralra = np.std(sim_mean_ralra)
ste_ralra = std_ralra/np.sqrt(amount_of_sim)

mean_u2alra = np.mean(u2alra)
std_u2alra = np.std(u2alra)
ste_u2alra = std_u2alra/np.sqrt(amount_of_sim)


###########################################################################################################################
#PLOTTING
###########################################################################################################################
matplotlib.rcParams['font.family'] = 'times new roman'
matplotlib.rcParams['font.size'] = 14 

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = "2Constant_lr"
save_dir = os.path.join(script_dir, folder_name)

figsize = (5,7)
random.seed(5454654865)
np.random.seed(569698)

#Figure 1: REWARDS GROUPED PER EPSILON VALUE
#plot rewards for each situation
#first bars are grouped per epsilon value
groups = [fr'$\alpha$ = {lr_sta} (optimal for stable)', fr'$\alpha$ = {lr_vol} (optimal for volatile)', fr'$\alpha$ = {lr_adv} (optimal for adversarial)']
categories = ['stable environment', 'volatile environment', 'adversarial environment']
values = np.array([[mean_rslrs, mean_rvlrs, mean_ralrs],
                   [mean_rslrv, mean_rvlrv, mean_ralrv],
                   [mean_rslra, mean_rvlra, mean_ralra]])
errors = np.array([[ste_rslrs, ste_rvlrs, ste_ralrs],
                   [ste_rslrv, ste_rvlrv, ste_ralrv],
                   [ste_rslra, ste_rvlra, ste_ralra]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    # Plot the bars for the category
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], edgecolor = 'red', linewidth=1.3,alpha=0.5, label=f'{category}' if i == 0 else "")

# Customize the plot
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Reward')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
ax.legend()
fig_name = os.path.join(save_dir, f'reward_per_lr')
plt.savefig(fig_name)
plt.show()


#Figure 2: REWARDS GROUPED PER ENVIRONMENT
#second bars are grouped per environment
categories = [fr'$\alpha$ = {lr_sta} (optimal for stable)', fr'$\alpha$ = {lr_vol} (optimal for volatile)', fr'$\alpha$ = {lr_adv} (optimal for adversarial)']
groups = ['stable environment', 'volatile environment', 'adversarial environment']

values = np.array([[mean_rslrs,mean_rslrv,mean_rslra],
                  [mean_rvlrs,mean_rvlrv,mean_rvlra],
                  [mean_ralrs,mean_ralrv,mean_ralra]])


errors = np.array([[ste_rslrs,ste_rslrv,ste_rslra],
                  [ste_rvlrs,ste_rvlrv,ste_rvlra],
                  [ste_ralrs,ste_ralrv,ste_ralra]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    # Plot the bars for the category
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], edgecolor = 'red', linewidth=1.3,alpha=0.5, label=f'{category}' if i == 0 else "")

# Customize the plot
ax.set_xlabel(r'$\alpha$')

ax.set_ylabel('Reward')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
ax.legend()
fig_name = os.path.join(save_dir, f'reward_per_env')
plt.savefig(fig_name)
plt.show()



#Figure 3: OPTIMAL U-VALUES
#plot u-values
#optimal u-values
categories = ['stable', 'volatile', 'adversarial']
values = [mean_u2slrs, mean_u2vlrv, mean_u2alra]
errors = [ste_u2slrs, ste_u2vlrv, ste_u2alra]
fig, ax = plt.subplots(figsize=figsize)
ax.bar(categories, values, color=category_colors, edgecolor='red', alpha=0.75, yerr=errors, capsize=5)
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
values = np.array([[mean_u2slrs, mean_u2vlrs, mean_u2alrs],
                   [mean_u2slrv, mean_u2vlrv, mean_u2alrv],
                   [mean_u2slra, mean_u2vlra, mean_u2alra]])

errors = np.array([[ste_u2slrs, ste_u2vlrs, ste_u2alrs],
                   [ste_u2slrv, ste_u2vlrv, ste_u2alrv],
                   [ste_u2slra, ste_u2vlra, ste_u2alra]])

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
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('U-values')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
ax.legend()
fig_name = os.path.join(save_dir, f'ru_values_all')
plt.savefig(fig_name)
plt.show()



#Figure 5: AVERAGE REWARDS ACROSS ENVIRONMENTS
#plot rewards as an average across three environments
R_lrs = np.concatenate((rslrs, rvlrs, ralrs), axis=0)
R_lrv = np.concatenate((rslrv, rvlrv, ralrv), axis=0)
R_lra = np.concatenate((rslra, rvlra, ralra), axis=0)
RML = np.concatenate((rslrs, rvlrv, ralra), axis=0)

sim_mean_R_lrs = np.mean(R_lrs, axis=1)
mean_R_lrs = np.mean(sim_mean_R_lrs)
std_R_lrs = np.std(sim_mean_R_lrs)
ste_R_lrs = std_R_lrs/np.sqrt(3*amount_of_sim)

sim_mean_R_lrv = np.mean(R_lrv, axis=1)
mean_R_lrv = np.mean(sim_mean_R_lrv)
std_R_lrv = np.std(sim_mean_R_lrv)
ste_R_lrv = std_R_lrv/np.sqrt(3*amount_of_sim)

sim_mean_R_lra = np.mean(R_lra, axis=1)
mean_R_lra = np.mean(sim_mean_R_lra)
std_R_lra = np.std(sim_mean_R_lra)
ste_R_lra = std_R_lra/np.sqrt(3*amount_of_sim)

sim_mean_RML = np.mean(RML, axis=1)
mean_RML = np.mean(sim_mean_RML)
std_RML = np.std(sim_mean_RML)
ste_RML = std_RML/np.sqrt(3*amount_of_sim)



categories = [r'stable-optimal $\alpha$', r'volatile-optimal $\alpha$', r'adversarial-optimal $\alpha$', r'environment-specific $\alpha$']
values = [mean_R_lrs, mean_R_lrv, mean_R_lra, mean_RML]
errors = [ste_R_lrs, ste_R_lrv, ste_R_lra, ste_RML]

# Create the bar plot with error bars
fig, ax = plt.subplots(figsize=figsize)
ax.bar(categories, values, color='red', edgecolor='black', linewidth=1.2, alpha=0.5, yerr=errors, capsize=5)

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
print('OPT LR DIFFERENCES BETWEEN REWARDS')
#check normality:
_, p1 = stats.shapiro(sim_mean_R_lrs)
_, p2 = stats.shapiro(sim_mean_R_lrv)
_, p3 = stats.shapiro(sim_mean_R_lra)
_, p4 = stats.shapiro(sim_mean_RML)


#check for equal variances:
_, p = stats.levene(sim_mean_R_lrs, sim_mean_R_lrv, sim_mean_R_lra, sim_mean_RML)


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
other_groups = [sim_mean_R_lrs, sim_mean_R_lrv, sim_mean_R_lra]
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
    print(f"Comparison env-specific LR vs {group_name} is {result} (Corrected P-value: {p_corr})")


#######################################
#Difference between average U-values across environments?
#######################################
print('OPT LR DIFFERENCES BETWEEN U-VALUES')
#for normally distributed but unequal variances

#check normality:
_, p1 = stats.shapiro(u2slrs)
_, p2 = stats.shapiro(u2vlrv)
_, p3 = stats.shapiro(u2alra)

#check for equal variances:
_, p = stats.levene(u2slrs, u2vlrv, u2alra)

print(f"P-values for normality: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}")
print(f"P-value for equal variances: {p}")

if p1 < 0.05 or p2 < 0.05 or p3 < 0.05:
    normality = False
else: normality = True

if p < 0.05 :
    equal_variances = False
else: equal_variances = True

print(f"P-values for normality: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}, and so normality is {normality}")
print(f"P-value for equal variances: {p}, and so equal_variance is {equal_variances}")


# Assuming data is in arrays group1, group2, and group3
groups = [u2slrs, u2vlrv, u2alra]
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


