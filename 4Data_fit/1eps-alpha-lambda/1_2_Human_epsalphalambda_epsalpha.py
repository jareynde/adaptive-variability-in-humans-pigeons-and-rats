#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Human data in seperate files per subject and condition
Model one: eps, alpha, lambda
Model two: eps, alpha
"""
import time
start = time.time()
from scipy.optimize import minimize # finding optimal params in models
from scipy import stats             # statistical tools
import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import matplotlib.pyplot as plt     # plotting
from scipy.optimize import differential_evolution # finding optimal params in models
import pickle
import random




#Frist: Read Neuringer's data
data_dir1 = os.path.dirname(os.path.abspath(__file__))
data_dir2 = os.path.join(data_dir1, 'ALL_DATA')
data_dir = os.path.join(data_dir2, 'Human')

dir_sub1con2 = os.path.join(data_dir, 'sub1con2.xlsx')
Sub1_data2 = pd.read_excel(dir_sub1con2)
sub1_data2 = Sub1_data2.to_numpy()

dir_sub1con4 = os.path.join(data_dir, 'sub1con4.xlsx')
Sub1_data4 = pd.read_excel(dir_sub1con4)
sub1_data4 = Sub1_data4.to_numpy()

dir_sub1con8 = os.path.join(data_dir, 'sub1con8.xlsx')
Sub1_data8 = pd.read_excel(dir_sub1con8)
sub1_data8 = Sub1_data8.to_numpy()


dir_sub2con2 = os.path.join(data_dir, 'sub2con2.xlsx')
Sub2_data2 = pd.read_excel(dir_sub2con2)
sub2_data2 = Sub2_data2.to_numpy()

dir_sub2con4 = os.path.join(data_dir, 'sub2con4.xlsx')
Sub2_data4 = pd.read_excel(dir_sub2con4)
sub2_data4 = Sub2_data4.to_numpy()

dir_sub2con8 = os.path.join(data_dir, 'sub2con8.xlsx')
Sub2_data8 = pd.read_excel(dir_sub2con8)
sub2_data8 = Sub2_data8.to_numpy()



dir_sub3con2 = os.path.join(data_dir, 'sub3con2.xlsx')
Sub3_data2 = pd.read_excel(dir_sub3con2)
sub3_data2 = Sub3_data2.to_numpy()

dir_sub3con4 = os.path.join(data_dir, 'sub3con4.xlsx')
Sub3_data4 = pd.read_excel(dir_sub3con4)
sub3_data4 = Sub3_data4.to_numpy()

dir_sub3con8 = os.path.join(data_dir, 'sub3con8.xlsx')
Sub3_data8 = pd.read_excel(dir_sub3con8)
sub3_data8 = Sub3_data8.to_numpy()



dir_sub4con2 = os.path.join(data_dir, 'sub4con2.xlsx')
Sub4_data2 = pd.read_excel(dir_sub4con2)
sub4_data2 = Sub4_data2.to_numpy()

dir_sub4con4 = os.path.join(data_dir, 'sub4con4.xlsx')
Sub4_data4 = pd.read_excel(dir_sub4con4)
sub4_data4 = Sub4_data4.to_numpy()

dir_sub4con8 = os.path.join(data_dir, 'sub4con8.xlsx')
Sub4_data8 = pd.read_excel(dir_sub4con8)
sub4_data8 = Sub4_data8.to_numpy()



dir_sub5con2 = os.path.join(data_dir, 'sub5con2.xlsx')
Sub5_data2 = pd.read_excel(dir_sub5con2)
sub5_data2 = Sub5_data2.to_numpy()

dir_sub5con4 = os.path.join(data_dir, 'sub5con4.xlsx')
Sub5_data4 = pd.read_excel(dir_sub5con4)
sub5_data4 = Sub5_data4.to_numpy()

dir_sub5con8 = os.path.join(data_dir, 'sub5con8.xlsx')
Sub5_data8 = pd.read_excel(dir_sub5con8)
sub5_data8 = Sub5_data8.to_numpy()



dir_sub6con2 = os.path.join(data_dir, 'sub6con2.xlsx')
Sub6_data2 = pd.read_excel(dir_sub6con2)
sub6_data2 = Sub6_data2.to_numpy()

dir_sub6con4 = os.path.join(data_dir, 'sub6con4.xlsx')
Sub6_data4 = pd.read_excel(dir_sub6con4)
sub6_data4 = Sub6_data4.to_numpy()

dir_sub6con8 = os.path.join(data_dir, 'sub6con8.xlsx')
Sub6_data8 = pd.read_excel(dir_sub6con8)
sub6_data8 = Sub6_data8.to_numpy()
print('data is', sub6_data8)






#Second, define the negative log likelihood function
#model 1
#alpha, eps, unchosen = params
def negll_RW_eps_alpha_uc(params, k, r):

    alpha, eps, unchosen = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)
    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k

        # update Q values for chosen option:
        Q_k[np.arange(len(Q_k)) != k[t]] += unchosen

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model2
def negll_RW_eps_alpha(params, k, r):

    alpha, eps = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)
    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += alpha * delta_k



    negLL = -np.sum(np.log(choice_prob)) 

    return negLL
    




Q_int = 1



strategy = 'randtobest1bin'

rewards = {}
mean_rewards = np.zeros(18)
standarderror_rewards = np.zeros(18)

results1 = {}
data_eps1 = np.zeros(18, dtype=float)
data_alpha1 = np.zeros(18, dtype=float)
data_unchosen1 = np.zeros(18, dtype=float)
data_BIC1 = np.zeros(18, dtype=float)
data_negLL1 = np.zeros(18, dtype=float)
data_trials1 = np.zeros(18, dtype=float)


results2 = {}
data_eps2 = np.zeros(18, dtype=float)
data_alpha2 = np.zeros(18, dtype=float)
data_BIC2 = np.zeros(18, dtype=float)
data_negLL2 = np.zeros(18, dtype=float)
data_trials2 = np.zeros(18, dtype=float)



choices = np.zeros(18,dtype=float)



for sub, sub_data in enumerate([sub1_data2, sub2_data2, sub3_data2, sub4_data2, sub5_data2, sub6_data2, sub1_data4, sub2_data4, sub3_data4, sub4_data4, sub5_data4, sub6_data4, sub1_data8, sub2_data8, sub3_data8, sub4_data8, sub5_data8, sub6_data8]):


    print('starting at sub', {sub})
    trials = np.shape(sub_data)[0]
    print('trials is',trials)
    k= sub_data[:,3]
    k= np.add(k,-1)
    k= k.astype(int)
            
    r= sub_data[:,4]
    r= r.astype(int)
    
    K = np.max(k)+1

    
    bounds1=[(0,1), (0,1), (-1/K,1/K)]
    bounds2=[(0,1), (0,1)]

    choices[sub] = sub_data[10,1]
    seedings = int(np.sum(k[5:10])*trials)
    print('random seed for', sub, 'is', seedings)

    random.seed(seedings)
    np.random.seed(seedings)

    #fit model 1: RW, fitting eps, alpha and lambda unchosen
    result1 = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
    negLL1 = result1.fun
    param_fits1 = result1.x

    data_alpha1[sub] = param_fits1[0]
    data_eps1[sub] = param_fits1[1]
    data_unchosen1[sub] = param_fits1[2]
    data_negLL1[sub] = negLL1
    data_trials1[sub] = trials

    BIC1 = len(bounds1) * np.log(trials) + 2*negLL1
    data_BIC1[sub] = BIC1
    print('model 1 done')


    #fit model 2: RW, fitting eps and alpha, no lambda unchosen
    result2 = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
    negLL2 = result2.fun
    param_fits2 = result2.x

    data_alpha2[sub] = param_fits2[0]
    data_eps2[sub] = param_fits2[1]
    data_negLL2[sub] = negLL2
    data_trials2[sub] = trials

    BIC2 = len(bounds2) * np.log(trials) + 2*negLL2
    data_BIC2[sub] = BIC2
    print('model 2 done')



   
results1['eps'] = data_eps1
results1[f'alpha'] = data_alpha1
results1[f'unchosen'] = data_unchosen1
results1[f'BIC'] = data_BIC1
results1[f'negLL'] = data_negLL1
results1[f'trials'] = data_trials1
results1['Choices'] = choices
    
results2['eps'] = data_eps2
results2[f'alpha'] = data_alpha2
results2[f'BIC'] = data_BIC2
results2[f'negLL'] = data_negLL2
results2[f'trials'] = data_trials2
results2['Choices'] = choices





BIC = {}
BIC['eps-alpha-uc'] = data_BIC1
BIC['eps-alpha'] = data_BIC2



save_dir = os.path.join(data_dir1, 'Human_eps-alpha-lambda')

dict_name = f'1Human_eps-alpha-lambda.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results1)
df.to_excel(dict_save)


dict_name = f'2Human_eps-alpha.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results2)
df.to_excel(dict_save)

dict_name = f'Human_BIC.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=BIC)
df.to_excel(dict_save)