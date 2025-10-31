#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Model one: eps, alpha lambda
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
    

data_dir1 = os.path.dirname(os.path.abspath(__file__))
data_dir2 = os.path.join(data_dir1, 'ALL_DATA')
data_dir3 = os.path.join(data_dir2, 'rats')

file_dir = os.path.join(data_dir3,'V4_usable')

maps = os.listdir(file_dir)

Q_int = 1


strategy = 'randtobest1bin'


rat_nr = np.array([])
file_nr = np.array([])

all_eps1 = np.array([]) 
all_alpha1 = np.array([]) 
all_unchosen1 = np.array([]) 
all_BIC1 = np.array([]) 
all_negLL1 = np.array([]) 
all_trials1 = np.array([]) 

av_eps1 = np.array([]) 
av_alpha1 = np.array([]) 
av_unchosen1 = np.array([]) 
av_BIC1 = np.array([]) 
av_negLL1 = np.array([]) 
av_trials1 = np.array([])
results1 = {}




all_eps2 = np.array([]) 
all_alpha2 = np.array([]) 
all_BIC2 = np.array([]) 
all_negLL2 = np.array([]) 
all_trials2 = np.array([]) 

av_eps2 = np.array([]) 
av_alpha2 = np.array([]) 
av_BIC2 = np.array([]) 
av_negLL2 = np.array([]) 
av_trials2 = np.array([])
results2 = {}




for count, folder in enumerate(maps):
    print('starting rat', count)
    rat = 0
    count=count+1
    subj_dir = os.path.join(file_dir,f'{folder}')
    subj_data_list = os.listdir(subj_dir)

    sub_eps1 = np.array([]) 
    sub_alpha1 = np.array([]) 
    sub_unchosen1 = np.array([]) 
    sub_BIC1 = np.array([]) 
    sub_negLL1 = np.array([]) 
    sub_trials1 = np.array([])


    sub_eps2 = np.array([]) 
    sub_alpha2 = np.array([]) 
    sub_BIC2 = np.array([]) 
    sub_negLL2 = np.array([]) 
    sub_trials2 = np.array([])



    for file_name in subj_data_list:
        print('rat', count, 'file', file_name)

        rat_nr = np.append(rat_nr, count)
        file_nr = np.append(file_nr, file_name)

        data_dir = os.path.join(subj_dir, file_name)
        Data = pd.read_excel(data_dir)
        data = Data.to_numpy()
        
        trials = np.shape(data)[0]

        k= data[:,0]
        k= np.add(k,-1)
        k= k.astype(int)
                
        r= data[:,2]
        r= r.astype(int)

        K = np.max(k)+1
        print('K is', K)
        bounds1=[(0,1), (0,1), (-1/K,1/K)]
        bounds2=[(0,1), (0,1)]

        seedings = int(np.sum(k[5:16])*np.sum(r[16:52])*(count+1))
        print('random seed for', count, 'is', seedings)

        random.seed(seedings)
        np.random.seed(seedings)


        #fit model 1: RW, fitting eps, alpha and lambda unchosen
        result1 = differential_evolution(negll_RW_eps_alpha_uc, bounds=bounds1, args=(k,r), strategy=strategy)
        negLL1 = result1.fun
        param_fits1 = result1.x
        BIC1 = len(bounds1) * np.log(trials) + 2*negLL1


        #this one saves al parameters of all subjects and all days
        all_eps1 = np.append(all_eps1, param_fits1[1])
        all_alpha1 = np.append(all_alpha1, param_fits1[0])
        all_unchosen1 = np.append(all_unchosen1, param_fits1[2])
        all_BIC1 = np.append(all_BIC1, BIC1)
        all_negLL1 = np.append(all_negLL1, negLL1)
        all_trials1 = np.append(all_trials1, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps1 = np.append(sub_eps1, param_fits1[1])
        sub_alpha1 = np.append(sub_alpha1, param_fits1[0])
        sub_unchosen1 = np.append(sub_unchosen1, param_fits1[2])
        sub_BIC1 = np.append(sub_BIC1, BIC1)
        sub_negLL1 = np.append(sub_negLL1, negLL1)
        sub_trials1 = np.append(sub_trials1, trials)

        print('model 1 done')





        #fit model 2: RW, fitting eps and alpha
        result2 = differential_evolution(negll_RW_eps_alpha, bounds=bounds2, args=(k,r), strategy=strategy)
        negLL2 = result2.fun
        param_fits2 = result2.x
        BIC2 = len(bounds2) * np.log(trials) + 2*negLL2


        #this one saves al parameters of all subjects and all days
        all_eps2 = np.append(all_eps2, param_fits2[1])
        all_alpha2 = np.append(all_alpha2, param_fits2[0])
        all_BIC2 = np.append(all_BIC2, BIC2)
        all_negLL2 = np.append(all_negLL2, negLL2)
        all_trials2 = np.append(all_trials2, trials)

        #this one saves parameters per subjects for all days; later an average parameter estimate is taken per subject over all days
        sub_eps2 = np.append(sub_eps2, param_fits2[1])
        sub_alpha2 = np.append(sub_alpha2, param_fits2[0])
        sub_BIC2 = np.append(sub_BIC2, BIC2)
        sub_negLL2 = np.append(sub_negLL2, negLL2)
        sub_trials2 = np.append(sub_trials2, trials)


        print('model 2 done')










    sub_av_eps1 = np.mean(sub_eps1)
    sub_av_alpha1 = np.mean(sub_alpha1)
    sub_av_unchosen1 = np.mean(sub_unchosen1)
    sub_av_BIC1 = np.mean(sub_BIC1)
    sub_av_negLL1 = np.mean(sub_negLL1)
    sub_av_trials1 = np.mean(sub_trials1)

    sub_av_eps2 = np.mean(sub_eps2)
    sub_av_alpha2 = np.mean(sub_alpha2)
    sub_av_BIC2 = np.mean(sub_BIC2)
    sub_av_negLL2 = np.mean(sub_negLL2)
    sub_av_trials2 = np.mean(sub_trials2)






    av_eps1 = np.append(av_eps1, sub_av_eps1)
    av_alpha1 = np.append(av_alpha1, sub_av_alpha1)
    av_unchosen1 = np.append(av_unchosen1, sub_av_unchosen1)
    av_BIC1 = np.append(av_BIC1, sub_av_BIC1)
    av_negLL1 = np.append(av_negLL1, sub_av_negLL1)
    av_trials1= np.append(av_trials1, sub_av_trials1)

    av_eps2 = np.append(av_eps2, sub_av_eps2)
    av_alpha2 = np.append(av_alpha2, sub_av_alpha2)
    av_BIC2 = np.append(av_BIC2, sub_av_BIC2)
    av_negLL2 = np.append(av_negLL2, sub_av_negLL2)
    av_trials2 = np.append(av_trials2, sub_av_trials2)



save_dir = os.path.join(data_dir1, 'Rat_eps-alpha-lambda')


all_results1 = {}
all_results2 = {}



all_results1[f'eps'] = all_eps1
all_results1[f'alpha'] = all_alpha1
all_results1[f'unchosen'] = all_unchosen1
all_results1[f'BIC'] = all_BIC1
all_results1[f'negLL'] = all_negLL1
all_results1[f'trials'] = all_trials1  
all_results1['rat'] = rat_nr
all_results1['file'] = file_nr

results1[f'eps'] = av_eps1
results1[f'alpha'] = av_alpha1
results1[f'unchosen'] = av_unchosen1
results1[f'BIC'] = av_BIC1
results1[f'negLL'] = av_negLL1
results1[f'trials'] = av_trials1


dict_name = f'1Rats_av_eps-alpha-lambda.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results1)
df.to_excel(dict_save)

dict_name = f'1Rats_all_eps-alpha-lambda.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results1)
df.to_excel(dict_save)




results2[f'eps'] = av_eps2
results2[f'alpha'] = av_alpha2
results2[f'BIC'] = av_BIC2
results2[f'negLL'] = av_negLL2
results2[f'trials'] = av_trials2


all_results2[f'eps'] = all_eps2
all_results2[f'alpha'] = all_alpha2
all_results2[f'BIC'] = all_BIC2
all_results2[f'negLL'] = all_negLL2
all_results2[f'trials'] = all_trials2  
all_results2['rat'] = rat_nr
all_results2['file'] = file_nr

dict_name = f'2Rats_av_eps-alpha.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=results2)
df.to_excel(dict_save)

dict_name = f'2Rats_all_eps-alpha.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=all_results2)
df.to_excel(dict_save)


BIC_all_data = {}
BIC_all_data['eps-alpha-uc'] = all_BIC1
BIC_all_data['eps-alpha'] = all_BIC2




BIC_av_data = {}
BIC_av_data['eps-alpha-uc'] = av_BIC1
BIC_av_data['eps-alpha'] = av_BIC2



dict_name = f'Rats_all_BIC.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=BIC_all_data)
df.to_excel(dict_save)

dict_name = f'Rats_av_BIC.xlsx'
dict_save = os.path.join(save_dir,dict_name)
df = pd.DataFrame(data=BIC_av_data)
df.to_excel(dict_save)