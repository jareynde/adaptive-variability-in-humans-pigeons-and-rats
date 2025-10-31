#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
from matplotlib.patches import Rectangle




##############################################################################################################
#Functions of Rescorla-Wagner model, dfferent environments
##############################################################################################################
#simulation of Rescorla-Wagner model in an adversarial context
def RW_adversarial(Q_alpha, eps, unchosen, T, Q_int):
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #unchosen   --->        unchosen value-bias
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=4 #the amount of choice options
    K_seq = K*K #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(K) for b in range(K)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    potential_reward = np.zeros((T,K))
    
    
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

            for j in range(K):
                current_seq = [k[t-1],j]
                current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
                current_freq = Freq1[current_index]
                if current_freq < np.percentile(Freq1,60):
                    potential_reward[t,j] = 1
                else: potential_reward[t,j] = 0

            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(K_seq, dtype=float)*(-1/(K_seq-1))
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/(K_seq-1))
            Freq1 = Freq1*0.984
            
            


         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, potential_reward, Freq1

matplotlib.rcParams['font.family'] = 'times new roman'
fontsize=18
K=4

k, potential_reward, Freq1 = RW_adversarial(Q_alpha=0.5, eps=0.5, unchosen= 0, T=1000, Q_int =1)
print(Freq1)
data = potential_reward[900:,:]

highlight = k[900:]  # 1x100 array, values 0.7

colors = ["#EBC313","#E70D1F","#5073E7","#53AF4A","#6B03C0","#FF9100","#2B0BB8","#FF07A0DF"]


colored_data = np.ones((4, 100, 4))
for row in range(K):
    mask = data[:, row] == 1
    for x in np.where(mask)[0]:
        colored_data[row, x, :] = plt.matplotlib.colors.to_rgba(colors[row])

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(colored_data, aspect="auto", origin="lower")

for i, row in enumerate(highlight):
    rect = Rectangle((i-0.5, row-0.5), 1, 1, fill=False,
                     edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)

ax.set_xticks(np.arange(0, data.shape[0]))
labels = [str(i+1) if (i+1) % 10 == 0 or (i+1) == 1 else "" for i in range(data.shape[0])]
ax.set_xticklabels(labels)

ax.set_yticks(np.arange(K))
ax.set_yticklabels([f"Action {i+1}" for i in range(K)])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
#plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
time = np.linspace(1, 100, 100, endpoint=True)
rewards = np.zeros(np.shape(data))
for i in range(K):
    rewards[:,i] = data[:,i]*0.5+(i/25)
    reward = data[:,i]*0.5+(i/25)
    ax.scatter(time, reward, color = colors[i])
print(np.shape(rewards), np.shape(time))
for i, row in enumerate(highlight):
    ax.scatter(i+1, rewards[i,row], edgecolor='black', color = colors[row])

plt.show()

'''

fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#EBC313","#E70D1F","#2EACBD","#FFAE00","#6B03C0","#1E6B17","#2B0BB8","#63E43BC1"]

# Loop over columns (8 rows to plot)
for row in range(data.shape[1]):
    # find indices where the value is 1
    x = np.where(data[:, row] == 1)[0] + 1  # +1 so x-ticks start at 1
    # draw bars at those positions
    ax.bar(x, [1]*len(x), bottom=row, width=1.0, color=colors[row], align='center', edgecolor='black')

ax.set_xticks(np.arange(1, data.shape[0]+1))  # ticks 1..100
ax.set_xlim(0.5, data.shape[0]+0.5)
ax.set_ylim(0, data.shape[1])
ax.set_yticks(np.arange(data.shape[1]) + 0.5)
ax.set_yticklabels([f"Action {i+1}" for i in range(data.shape[1])])
labels = [str(i) if i % 10 == 0 or i == 1 else "" for i in range(1, data.shape[0]+1)]
ax.set_xticklabels(labels)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

'''




#simulation of Rescorla-Wagner model in stable context
def RW_stable(Q_alpha, eps, unchosen, T, Q_int):
    #alpha              --->        learning rate
    #eps                --->        epsilon
    #unchosen           --->        unchosen value-bias
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward_stable      --->        probabilites to recieve a reward
    
    K=4 #the amount of choice options    

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    potential_reward = np.zeros((T,K))

    reward_stable = [0.3,0.7,0.3,0.3]#,0.7,0.7,0.3,0.3]
    
    
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

        for j in range(K):
            a1 = reward_stable[j]
            a0 = 1-a1
            potential_reward[t,j] = np.random.choice([1, 0], p=[a1, a0])


         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, potential_reward


k, potential_reward = RW_stable(Q_alpha=0.1, eps=0.1, unchosen=0, T=10000, Q_int =1)
data = potential_reward[5000:5100]
highlight = k[5000:5100]
fig, ax = plt.subplots(figsize=(12, 6))


# Loop over columns (8 rows to plot)
for row in range(data.shape[1]):
    # find indices where the value is 1
    x = np.where(data[:, row] == 1)[0] + 1  # +1 so x-ticks start at 1
    # draw bars at those positions
    ax.bar(x, [1]*len(x), bottom=row, width=1.0, color=colors[row], align='center')

ax.set_xticks(np.arange(1, data.shape[0]+1))  # ticks 1..100
ax.set_xlim(0.5, data.shape[0]+0.5)
ax.set_ylim(0, data.shape[1])
ax.set_yticks(np.arange(data.shape[1]) + 0.5)
ax.set_yticklabels([f"Action {i+1}" for i in range(data.shape[1])])
labels = [str(i) if i % 10 == 0 or i == 1 else "" for i in range(1, data.shape[0]+1)]
ax.set_xticklabels(labels)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
time = np.linspace(1, 100, 100, endpoint=True)
rewards = np.zeros(np.shape(data))
for i in range(K):
    rewards[:,i] = data[:,i]*0.5+(i/25)
    reward = data[:,i]*0.5+(i/25)
    ax.scatter(time, reward, color = colors[i])
for i, row in enumerate(highlight):
    ax.scatter(i, rewards[i,row], edgecolor='black', color = colors[row])

plt.show()

#simulation of Rescorla-Wagner model in volatile context
def RW_volatile8(Q_alpha, eps, unchosen, T, Q_int):
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
    reward_volatile = [0.9,0.1,0.1,0.1,0.9,0.1,0.9,0.1]
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000]     

    potential_reward = np.zeros((T,K))

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
            for i in range(K):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        for j in range(K):
            a1 = reward_volatile[j]
            a0 = 1-a1
            potential_reward[t,j] = np.random.choice([1, 0], p=[a1, a0])



         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, potential_reward



def RW_volatile(Q_alpha, eps, unchosen, T, Q_int):
    #alpha              --->        learning rate
    #eps                --->        epsilon
    #unchosen           --->        unchosen value-bias
    #T                  --->        amount of trials for each simulation
    #Q_int              --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward_volatile    --->        probabilites to recieve a reward, shuffles among options

    K=4 #the amount of choice options
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards
    reward_volatile = [0.9,0.1,0.1,0.1]#,0.9,0.1,0.9,0.1]
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000]     

    potential_reward = np.zeros((T,K))

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
            new_max_index = index_vol[0]
            for i in range(K):
                if i == new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        for j in range(K):
            a1 = reward_volatile[j]
            a0 = 1-a1
            potential_reward[t,j] = np.random.choice([1, 0], p=[a1, a0])



         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    return k, potential_reward




k, potential_reward = RW_volatile(Q_alpha=0.8, eps=0.3, unchosen=0, T=10000, Q_int =1)
data = potential_reward[5000:5100]
highlight = k[5000:5100]



fig, ax = plt.subplots(figsize=(12, 6))


# Loop over columns (8 rows to plot)
for row in range(data.shape[1]):
    # find indices where the value is 1
    x = np.where(data[:, row] == 1)[0] + 1  # +1 so x-ticks start at 1
    # draw bars at those positions
    ax.bar(x, [1]*len(x), bottom=row, width=1.0, color=colors[row], align='center')

ax.set_xticks(np.arange(1, data.shape[0]+1))  # ticks 1..100
ax.set_xlim(0.5, data.shape[0]+0.5)
ax.set_ylim(0, data.shape[1])
ax.set_yticks(np.arange(data.shape[1]) + 0.5)
ax.set_yticklabels([f"Action {i+1}" for i in range(data.shape[1])])
labels = [str(i) if i % 10 == 0 or i == 1 else "" for i in range(1, data.shape[0]+1)]
ax.set_xticklabels(labels)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
time = np.linspace(1, 100, 100, endpoint=True)
rewards = np.zeros(np.shape(data))
for i in range(K):
    rewards[:,i] = data[:,i]*0.5+(i/25)
    reward = data[:,i]*0.5+(i/25)
    ax.scatter(time, reward, color = colors[i])
for i, row in enumerate(highlight):
    ax.scatter(i, rewards[i,row], edgecolor='black', color = colors[row])

plt.show()