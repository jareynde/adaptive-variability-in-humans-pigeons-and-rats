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



#Optimal epsilons in each environment
#From the ML_eps script: average of the last 100 epsilons for all ML simulations

eps_sta = 0.09
eps_vol = 0.24
eps_adv = 0.87
		
#0.868712752	0.093202183	0.236629337




Q_alpha = 0.44
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

kses = np.zeros([amount_of_sim, T])
rses = np.zeros([amount_of_sim, T])
ksev = np.zeros([amount_of_sim, T])
rsev = np.zeros([amount_of_sim, T])
ksea = np.zeros([amount_of_sim, T])
rsea = np.zeros([amount_of_sim, T])

kves = np.zeros([amount_of_sim, T])
rves = np.zeros([amount_of_sim, T])
kvev = np.zeros([amount_of_sim, T])
rvev = np.zeros([amount_of_sim, T])
kvea = np.zeros([amount_of_sim, T])
rvea = np.zeros([amount_of_sim, T])

kaes = np.zeros([amount_of_sim, T])
raes = np.zeros([amount_of_sim, T])
kaev = np.zeros([amount_of_sim, T])
raev = np.zeros([amount_of_sim, T])
kaea = np.zeros([amount_of_sim, T])
raea = np.zeros([amount_of_sim, T])


u1ses = np.zeros(amount_of_sim)
u2ses = np.zeros(amount_of_sim)
u1sev = np.zeros(amount_of_sim)
u2sev = np.zeros(amount_of_sim)
u1sea = np.zeros(amount_of_sim)
u2sea = np.zeros(amount_of_sim)

u1ves = np.zeros(amount_of_sim)
u2ves = np.zeros(amount_of_sim)
u1vev = np.zeros(amount_of_sim)
u2vev = np.zeros(amount_of_sim)
u1vea = np.zeros(amount_of_sim)
u2vea = np.zeros(amount_of_sim)

u1aes = np.zeros(amount_of_sim)
u2aes = np.zeros(amount_of_sim)
u1aev = np.zeros(amount_of_sim)
u2aev = np.zeros(amount_of_sim)
u1aea = np.zeros(amount_of_sim)
u2aea = np.zeros(amount_of_sim)


for sim in range(amount_of_sim):

    random.seed(seeds[sim])
    np.random.seed(seeds[sim])

    #in a stable environment
    kses[sim,:], rses[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps_sta, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable) #code: k stable (env) eps stable (optimal eps for stable)
    u1ses[sim], u2ses[sim] = u_value(seq=kses[sim,:], alternatives=8, two_alternatives=64)

    ksev[sim,:], rsev[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps_vol, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable)
    u1sev[sim], u2sev[sim] = u_value(seq=ksev[sim,:], alternatives=8, two_alternatives=64)

    ksea[sim,:], rsea[sim,:], Q_k_stored = RW_lambda_stable(Q_alpha=Q_alpha, eps=eps_adv, unchosen=unchosen, T=T, Q_int=Q_int, reward_stable=reward_stable)
    u1sea[sim], u2sea[sim] = u_value(seq=ksea[sim,:], alternatives=8, two_alternatives=64)

    random.seed(seeds[sim+amount_of_sim])
    np.random.seed(seeds[sim+amount_of_sim])

    #in a volatile environment
    kves[sim,:], rves[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps_sta, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1ves[sim], u2ves[sim] = u_value(seq=kves[sim,:], alternatives=8, two_alternatives=64)

    kvev[sim,:], rvev[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps_vol, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vev[sim], u2vev[sim] = u_value(seq=kvev[sim,:], alternatives=8, two_alternatives=64)

    kvea[sim,:], rvea[sim,:], Q_k_stored = RW_lambda_volatile(Q_alpha=Q_alpha, eps=eps_adv, unchosen=unchosen, T=T, Q_int=Q_int, reward_volatile=reward_volatile)
    u1vea[sim], u2vea[sim] = u_value(seq=kvea[sim,:], alternatives=8, two_alternatives=64)

    random.seed(seeds[sim+2*amount_of_sim])
    np.random.seed(seeds[sim+2*amount_of_sim])

    #in an adversarial environment
    kaes[sim,:], raes[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps_sta, unchosen=unchosen, T=T, Q_int=Q_int)
    u1aes[sim], u2aes[sim] = u_value(seq=kaes[sim,:], alternatives=8, two_alternatives=64)

    kaev[sim,:], raev[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps_vol, unchosen=unchosen, T=T, Q_int=Q_int)
    u1aev[sim], u2aev[sim] = u_value(seq=kaev[sim,:], alternatives=8, two_alternatives=64)

    kaea[sim,:], raea[sim,:], Q_k_stored = RW_lambda_adversarial(Q_alpha=Q_alpha, eps=eps_adv, unchosen=unchosen, T=T, Q_int=Q_int)
    u1aea[sim], u2aea[sim] = u_value(seq=kaea[sim,:], alternatives=8, two_alternatives=64)


    print('check',sim)




#Stable
sim_mean_rses = np.mean(rses, axis=1)
mean_rses = np.mean(sim_mean_rses)
std_rses = np.std(sim_mean_rses)
ste_rses = std_rses/np.sqrt(amount_of_sim)

mean_u2ses = np.mean(u2ses)
std_u2ses = np.std(u2ses)
ste_u2ses = std_u2ses/np.sqrt(amount_of_sim)


sim_mean_rsev = np.mean(rsev, axis=1)
mean_rsev = np.mean(sim_mean_rsev)
std_rsev = np.std(sim_mean_rsev)
ste_rsev = std_rsev/np.sqrt(amount_of_sim)

mean_u2sev = np.mean(u2sev)
std_u2sev = np.std(u2sev)
ste_u2sev = std_u2sev/np.sqrt(amount_of_sim)


sim_mean_rsea = np.mean(rsea, axis=1)
mean_rsea = np.mean(sim_mean_rsea)
std_rsea = np.std(sim_mean_rsea)
ste_rsea = std_rsea/np.sqrt(amount_of_sim)

mean_u2sea = np.mean(u2sea)
std_u2sea = np.std(u2sea)
ste_u2sea = std_u2sea/np.sqrt(amount_of_sim)




#Volatile
sim_mean_rves = np.mean(rves, axis=1)
mean_rves = np.mean(sim_mean_rves)
std_rves = np.std(sim_mean_rves)
ste_rves = std_rves/np.sqrt(amount_of_sim)

mean_u2ves = np.mean(u2ves)
std_u2ves = np.std(u2ves)
ste_u2ves = std_u2ves/np.sqrt(amount_of_sim)


sim_mean_rvev = np.mean(rvev, axis=1)
mean_rvev = np.mean(sim_mean_rvev)
std_rvev = np.std(sim_mean_rvev)
ste_rvev = std_rvev/np.sqrt(amount_of_sim)

mean_u2vev = np.mean(u2vev)
std_u2vev = np.std(u2vev)
ste_u2vev = std_u2vev/np.sqrt(amount_of_sim)


sim_mean_rvea = np.mean(rvea, axis=1)
mean_rvea = np.mean(sim_mean_rvea)
std_rvea = np.std(sim_mean_rvea)
ste_rvea = std_rvea/np.sqrt(amount_of_sim)

mean_u2vea = np.mean(u2vea)
std_u2vea = np.std(u2vea)
ste_u2vea = std_u2vea/np.sqrt(amount_of_sim)





#adversarial
sim_mean_raes = np.mean(raes, axis=1)
mean_raes = np.mean(sim_mean_raes)
std_raes = np.std(sim_mean_raes)
ste_raes = std_raes/np.sqrt(amount_of_sim)

mean_u2aes = np.mean(u2aes)
std_u2aes = np.std(u2aes)
ste_u2aes = std_u2aes/np.sqrt(amount_of_sim)


sim_mean_raev = np.mean(raev, axis=1)
mean_raev = np.mean(sim_mean_raev)
std_raev = np.std(sim_mean_raev)
ste_raev = std_raev/np.sqrt(amount_of_sim)

mean_u2aev = np.mean(u2aev)
std_u2aev = np.std(u2aev)
ste_u2aev = std_u2aev/np.sqrt(amount_of_sim)


sim_mean_raea = np.mean(raea, axis=1)
mean_raea = np.mean(sim_mean_raea)
std_raea = np.std(sim_mean_raea)
ste_raea = std_raea/np.sqrt(amount_of_sim)

mean_u2aea = np.mean(u2aea)
std_u2aea = np.std(u2aea)
ste_u2aea = std_u2aea/np.sqrt(amount_of_sim)

###########################################################################################################################
#PLOTTING
###########################################################################################################################
matplotlib.rcParams['font.family'] = 'times new roman'
matplotlib.rcParams['font.size'] = 14 

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = "2Constant_eps"
save_dir = os.path.join(script_dir, folder_name)

figsize = (5,7)
random.seed(5454654865)
np.random.seed(569698)


#Figure 1: REWARDS GROUPED PER EPSILON VALUE
#plot rewards for each situation
#first bars are grouped per epsilon value
groups = [fr'$\epsilon$ = {eps_sta} (optimal for stable)', fr'$\epsilon$ = {eps_vol} (optimal for volatile)', fr'$\epsilon$ = {eps_adv} (optimal for adversarial)']
categories = ['stable environment', 'volatile environment', 'adversarial environment']
values = np.array([[mean_rses, mean_rves, mean_raes],
                   [mean_rsev, mean_rvev, mean_raev],
                   [mean_rsea, mean_rvea, mean_raea]])
errors = np.array([[ste_rses, ste_rves, ste_raes],
                   [ste_rsev, ste_rvev, ste_raev],
                   [ste_rsea, ste_rvea, ste_raea]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']


fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], edgecolor = 'purple', linewidth=1.3,alpha=0.5, label=f'{category}' if i == 0 else "")

ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Reward')
plt.ylim([0,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
#ax.legend()
fig_name = os.path.join(save_dir, f'reward_per_eps')
plt.savefig(fig_name)
plt.show()

#Figure 2: REWARDS GROUPED PER ENVIRONMENT
#second bars are grouped per environment
categories = [fr'$\epsilon$ = {eps_sta} (optimal for stable)', fr'$\epsilon$ = {eps_vol} (optimal for volatile)', fr'$\epsilon$ = {eps_adv} (optimal for adversarial)']
groups = ['stable environment', 'volatile environment', 'adversarial environment']

values = np.array([[mean_rses,mean_rsev,mean_rsea],
                  [mean_rves,mean_rvev,mean_rvea],
                  [mean_raes,mean_raev,mean_raea]])

errors = np.array([[ste_rses,ste_rsev,ste_rsea],
                  [ste_rves,ste_rvev,ste_rvea],
                  [ste_raes,ste_raev,ste_raea]])

category_colors = ['darkcyan', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=figsize)

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    category_errors = errors[:, j]
    
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, yerr=category_errors[i], capsize=5, color=category_colors[j], edgecolor = 'purple', linewidth=1.3,alpha=0.5, label=f'{category}' if i == 0 else "")

ax.set_xlabel(r'$\epsilon$')
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
values = [mean_u2ses, mean_u2vev, mean_u2aea]
errors = [ste_u2ses, ste_u2vev, ste_u2aea]
fig, ax = plt.subplots(figsize=figsize)
ax.bar(categories, values,  color=category_colors, edgecolor='purple',alpha=0.75,linewidth=1.2, yerr=errors, capsize=5)
ax.set_xlabel('environment')
ax.set_ylabel('U-Values')
ax.set_ylim([0,1])
fig_name = os.path.join(save_dir, f'u_values_opt')
plt.savefig(fig_name)
plt.show()


#Figure 4: ALL U-VALUES
#all u_values
groups = [r'optimal for stable', r'optimal for volatile', r'optimal for adversarial']
categories = ['stable environment', 'volatile environment', 'adversarial environment']
values = np.array([[mean_u2ses, mean_u2ves, mean_u2aes],
                   [mean_u2sev, mean_u2vev, mean_u2aev],
                   [mean_u2sea, mean_u2vea, mean_u2aea]])

errors = np.array([[ste_u2ses, ste_u2ves, ste_u2aes],
                   [ste_u2sev, ste_u2vev, ste_u2aev],
                   [ste_u2sea, ste_u2vea, ste_u2aea]])

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

ax.set_xlabel(r'$\epsilon$')
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
R_es = np.concatenate((rses, rves, raes), axis=0)
R_ev = np.concatenate((rsev, rvev, raev), axis=0)
R_ea = np.concatenate((rsea, rvea, raea), axis=0)
RML = np.concatenate((rses, rvev, raea), axis=0)

sim_mean_R_es = np.mean(R_es, axis=1)
mean_R_es = np.mean(sim_mean_R_es)
std_R_es = np.std(sim_mean_R_es)
ste_R_es = std_R_es/np.sqrt(3*amount_of_sim)

sim_mean_R_ev = np.mean(R_ev, axis=1)
mean_R_ev = np.mean(sim_mean_R_ev)
std_R_ev = np.std(sim_mean_R_ev)
ste_R_ev = std_R_ev/np.sqrt(3*amount_of_sim)

sim_mean_R_ea = np.mean(R_ea, axis=1)
mean_R_ea = np.mean(sim_mean_R_ea)
std_R_ea = np.std(sim_mean_R_ea)
ste_R_ea = std_R_ea/np.sqrt(3*amount_of_sim)

sim_mean_RML = np.mean(RML, axis=1)
mean_RML = np.mean(sim_mean_RML)
std_RML = np.std(sim_mean_RML)
ste_RML = std_RML/np.sqrt(3*amount_of_sim)



categories = [r'stable-optimal $\epsilon$', r'volatile-optimal $\epsilon$', r'adversarial-optimal $\epsilon$', r'environment-specific $\epsilon$']
values = [mean_R_es, mean_R_ev, mean_R_ea, mean_RML]
errors = [ste_R_es, ste_R_ev, ste_R_ea, ste_RML]


# Create the bar plot with error bars
fig, ax = plt.subplots(figsize=figsize)
ax.bar(categories, values, color='purple', edgecolor='black', alpha=0.5, linewidth=1.2, yerr=errors, capsize=5)

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
print('OPT EPS DIFFERENCES BETWEEN REWARDS')
#check normality:
_, p1 = stats.shapiro(sim_mean_R_es)
_, p2 = stats.shapiro(sim_mean_R_ev)
_, p3 = stats.shapiro(sim_mean_R_ea)
_, p4 = stats.shapiro(sim_mean_RML)


#check for equal variances:
_, p = stats.levene(sim_mean_R_es, sim_mean_R_ev, sim_mean_R_ea, sim_mean_RML)


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
other_groups = [sim_mean_R_es, sim_mean_R_ev, sim_mean_R_ea]
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
    print(f"Comparison env-specific eps vs {group_name} is {result} (Corrected P-value: {p_corr})")


#######################################
#Difference between average U-values across environments?
#######################################
print('OPT EPS DIFFERENCES BETWEEN U-VALUES')
#check normality:
_, p1 = stats.shapiro(u2ses)
_, p2 = stats.shapiro(u2vev)
_, p3 = stats.shapiro(u2aea)

#check for equal variances:
_, p = stats.levene(u2ses, u2vev, u2aea)


if p1 < 0.05 or p2 < 0.05 or p3 < 0.05:
    normality = False
else: normality = True

if p < 0.05 :
    equal_variances = False
else: equal_variances = True

print(f"P-values for normality: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}, and so normality is {normality}")
print(f"P-value for equal variances: {p}, and so equal_variance is {equal_variances}")

# Assuming data is in arrays group1, group2, and group3
groups = [u2ses, u2vev, u2aea]
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


