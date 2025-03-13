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
Take results of the gridsearch and find the highest reward that was obtained and the parameter values that did it.
Also find the optimal parameter for each of our other two 'common sense' values, i.e.:
* optimal eps when lr=0.25 and lambda=0
* optimal lr when eps=0.3 and lambda=0
* optimal lambda when eps=0.3 and lr=0.25

EXACTLY THE SAME AS THE OTHER 2_constant_heatmap.py
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

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'reward')
heat = 51

all_eps = np.linspace(0, 1, heat, endpoint=True)
all_lr = np.linspace(0, 1, heat, endpoint=True)
all_unchosen = np.linspace(-1, 1, heat, endpoint=True)

#Take results of the gridsearch and find the highest reward that was obtained and the parameter values that did it.


max_stable_rewards = np.zeros([heat, 4], dtype=float) #for every eps value there is one row, columns contain: max reward value, eps, learning rate, lambda
max_volatile_rewards = np.zeros([heat, 4], dtype=float) #for every eps value there is one row, columns contain: max reward value, eps, learning rate, lambda
max_adversarial_rewards = np.zeros([heat, 4], dtype=float) #for every eps value there is one row, columns contain: max reward value, eps, learning rate, lambda


for i in range(heat):
    advtitle = f'adv_mr_eps{i}.xlsx'
    adv_dic = os.path.join(data_dir,advtitle)
    statitle = f'sta_mr_eps{i}.xlsx'
    sta_dic = os.path.join(data_dir,statitle)
    voltitle = f'vol_mr_eps{i}.xlsx'
    vol_dic = os.path.join(data_dir,voltitle)

    advdata = pd.read_excel(adv_dic, engine="openpyxl")
    advdata = advdata.to_numpy()  
    advdata = advdata[:,1:]

    stadata = pd.read_excel(sta_dic, engine="openpyxl")
    stadata = stadata.to_numpy()  
    stadata = stadata[:,1:]

    voldata = pd.read_excel(vol_dic, engine="openpyxl")
    voldata = voldata.to_numpy()
    voldata = voldata[:,1:]




    advmax = np.max(advdata)
    advmaxi = np.unravel_index(np.argmax(advdata), advdata.shape)
    max_adversarial_rewards[i,0] = advmax
    max_adversarial_rewards[i,1] = all_eps[i]
    max_adversarial_rewards[i,2] = all_lr[advmaxi[0]]
    max_adversarial_rewards[i,3] = all_unchosen[advmaxi[1]]


    stamax = np.max(stadata)
    stamaxi = np.unravel_index(np.argmax(stadata), stadata.shape)
    max_stable_rewards[i,0] = stamax
    max_stable_rewards[i,1] = all_eps[i]
    max_stable_rewards[i,2] = all_lr[stamaxi[0]]
    max_stable_rewards[i,3] = all_unchosen[stamaxi[1]]

    
    volmax = np.max(voldata)
    volmaxi = np.unravel_index(np.argmax(voldata), voldata.shape)
    max_volatile_rewards[i,0] = volmax
    max_volatile_rewards[i,1] = all_eps[i]
    max_volatile_rewards[i,2] = all_lr[volmaxi[0]]
    max_volatile_rewards[i,3] = all_unchosen[volmaxi[1]]




print(max_volatile_rewards)
advmax2 = np.max(max_adversarial_rewards[:,0])
advmaxi2 = np.argmax(max_adversarial_rewards[:,0])
advmax_eps = max_adversarial_rewards[advmaxi2,1]
advmax_lr = max_adversarial_rewards[advmaxi2,2]
advmax_lambda = max_adversarial_rewards[advmaxi2,3]

print('the highest reward in adversarial is', advmax2, 'where eps is', advmax_eps, 'learning rate is', advmax_lr, 'and lambda is', advmax_lambda ) 


stamax2 = np.max(max_stable_rewards[:,0])
stamaxi2 = np.argmax(max_stable_rewards[:,0])
stamax_eps = max_stable_rewards[stamaxi2,1]
stamax_lr = max_stable_rewards[stamaxi2,2]
stamax_lambda = max_stable_rewards[stamaxi2,3]

print('the highest reward in stable is', stamax2, 'where eps is', stamax_eps, 'learning rate is', stamax_lr, 'and lambda is', stamax_lambda ) 




volmax2 = np.max(max_volatile_rewards[:,0])
volmaxi2 = np.argmax(max_volatile_rewards[:,0])
volmax_eps = max_volatile_rewards[volmaxi2,1]
volmax_lr = max_volatile_rewards[volmaxi2,2]
volmax_lambda = max_volatile_rewards[volmaxi2,3]

print('the highest reward in volatile is', volmax2, 'where eps is', volmax_eps, 'learning rate is', volmax_lr, 'and lambda is', volmax_lambda ) 

'''

#Find the optimal parameter for each of our other two 'common sense' values

lr_index = np.where(all_lr == 0.24)[0][0]
lambda_index = np.where(all_unchosen == 0)[0][0]
eps_index = np.where(all_eps == 0.3)[0][0]

MLeps = np.zeros([heat, 3]) #3 columns, one for each environment
for i in range(heat):
    advtitle = f'adv_mr_eps{i}.xlsx'
    adv_dic = os.path.join(data_dir,advtitle)
    statitle = f'sta_mr_eps{i}.xlsx'
    sta_dic = os.path.join(data_dir,statitle)
    voltitle = f'vol_mr_eps{i}.xlsx'
    vol_dic = os.path.join(data_dir,voltitle)

    advdata = pd.read_excel(adv_dic, engine="openpyxl")
    advdata = advdata.to_numpy()  
    advdata = advdata[:,1:]

    stadata = pd.read_excel(sta_dic, engine="openpyxl")
    stadata = stadata.to_numpy()  
    stadata = stadata[:,1:]

    voldata = pd.read_excel(vol_dic, engine="openpyxl")
    voldata = voldata.to_numpy()
    voldata = voldata[:,1:]

    MLeps[i,0] = advdata[lr_index,lambda_index]
    MLeps[i,1] = stadata[lr_index,lambda_index]
    MLeps[i,2] = voldata[lr_index,lambda_index]


   


maxadv_epsi = np.argmax(MLeps[:,0])
maxadv_eps = all_eps[maxadv_epsi]

maxsta_epsi = np.argmax(MLeps[:,1])
maxsta_eps = all_eps[maxsta_epsi]

maxvol_epsi = np.argmax(MLeps[:,2])
maxvol_eps = all_eps[maxvol_epsi]


advtitle = f'adv_mr_eps{eps_index}.xlsx'
adv_dic = os.path.join(data_dir,advtitle)
statitle = f'sta_mr_eps{eps_index}.xlsx'
sta_dic = os.path.join(data_dir,statitle)
voltitle = f'vol_mr_eps{eps_index}.xlsx'
vol_dic = os.path.join(data_dir,voltitle)
advdata = pd.read_excel(adv_dic, engine="openpyxl")
advdata = advdata.to_numpy()  
advdata = advdata[:,1:]

stadata = pd.read_excel(sta_dic, engine="openpyxl")
stadata = stadata.to_numpy()  
stadata = stadata[:,1:]

voldata = pd.read_excel(vol_dic, engine="openpyxl")
voldata = voldata.to_numpy()
voldata = voldata[:,1:]



maxadv_lri = np.argmax(advdata[:,lambda_index])
maxadv_lr = all_lr[maxadv_lri]

maxsta_lri = np.argmax(stadata[:,lambda_index])
maxsta_lr = all_lr[maxsta_lri]

maxvol_lri = np.argmax(voldata[:,lambda_index])
maxvol_lr = all_lr[maxvol_lri]





maxadv_uci = np.argmax(advdata[lr_index,:])
maxadv_uc = all_unchosen[maxadv_uci]

maxsta_uci = np.argmax(stadata[lr_index,:])
maxsta_uc = all_unchosen[maxsta_uci]

maxvol_uci = np.argmax(voldata[lr_index,:])
maxvol_uc = all_unchosen[maxvol_uci]



print('best eps for common sense lr and lambda is', maxadv_eps, maxsta_eps, maxvol_eps)
print('best lr for common sense eps and lambda is', maxadv_lr, maxsta_lr, maxvol_lr)
print('best lambda for common sense lr and eps is', maxadv_uc, maxsta_uc, maxvol_uc)

'''