#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Human data in seperate files per subject and condition

In this copy, we fit these models:
*1: eps-lr

"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
from scipy.optimize import differential_evolution # finding optimal params in models
from scipy.optimize import differential_evolution, LinearConstraint # finding optimal params in models
from csv import DictWriter


#Define the negative log likelihood function
#negative loglikelihoods for each model
#model 1
def negll_RW_eps_lr(k):

    K = np.max(k)+1
    T = len(k)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, 1 / K)  # Start with probability eps/K for each option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)       

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


current_dict = script_dir = os.path.dirname(os.path.abspath(__file__))


#Human
#Read Neuringer's data
files = ['sub1con2.csv', 'sub2con2.csv', 'sub3con2.csv', 'sub4con2.csv', 'sub5con2.csv','sub6con2.csv', 'sub1con4.csv', 'sub2con4.csv', 'sub3con4.csv', 'sub4con4.csv', 'sub5con4.csv', 'sub6con4.csv', 'sub1con8.csv', 'sub2con8.csv', 'sub3con8.csv', 'sub4con8.csv', 'sub5con8.csv', 'sub6con8.csv']
Human_negLLall = np.zeros(len(files))
for sub, file in enumerate(files):

    #data_dir1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #data_dir2 = os.path.join(data_dir1, 'ALL_DATA_CSV')
    #data_dir = os.path.join(data_dir2, 'Human')
    #df = pd.read_csv(os.path.join(data_dir, file))

    df = pd.read_csv(os.path.join(current_dict, f"ALL_DATA_CSV/Human/{file}"))
    sub_data = df.to_numpy()

    trials = np.shape(sub_data)[0]
    k= sub_data[:,4]
    k= np.add(k,-1)
    k= k.astype(int)
            
    r= sub_data[:,5]
    r= r.astype(int)
    
    K = np.max(k)+1

    Human_negLLall[sub] = negll_RW_eps_lr(k)


#Pigeon
#Read Neuringer's data
files = ['sub1con2.csv', 'sub2con2.csv', 'sub3con2.csv', 'sub4con2.csv', 'sub5con2.csv', 'sub1con4.csv', 'sub2con4.csv', 'sub3con4.csv', 'sub4con4.csv', 'sub5con4.csv', 'sub1con8.csv', 'sub2con8.csv', 'sub3con8.csv', 'sub4con8.csv', 'sub5con8.csv']
Pigeon_negLLall = np.zeros(len(files))
for sub, file in enumerate(files):
    df = pd.read_csv(os.path.join(current_dict, f"ALL_DATA_CSV/Pigeon/{file}"))
    sub_data = df.to_numpy()

    trials = np.shape(sub_data)[0]
    k= sub_data[:,5]
    k= np.add(k,-1)
    k= k.astype(int)
            
    r= sub_data[:,6]
    r= r.astype(int)
    
    K = np.max(k)+1

    Pigeon_negLLall[sub] = negll_RW_eps_lr(k)


#Rat
species_folder = os.path.join(current_dict, f"ALL_DATA_CSV/Rats")
subj_folders = os.listdir(species_folder)

Rat_negLLall = np.zeros(len(subj_folders))
for count, folder in enumerate(subj_folders):
    count=count+1
    subj_dir = os.path.join(species_folder,f'{folder}')

    subj_data_list = os.listdir(subj_dir)

    sub_rat = np.array([]) 
    

    for file_name in subj_data_list:
        data_dir = os.path.join(subj_dir, file_name)
        Data = pd.read_csv(data_dir)
        data = Data.to_numpy()
        
        trials = np.shape(data)[0]

        k= data[:,1]
        k= np.add(k,-1)
        k= k.astype(int)
                
        r= data[:,3]
        r= r.astype(int)

        K = np.max(k)+1

        negLL = negll_RW_eps_lr(k)
        sub_rat = np.append(sub_rat, negLL)

    Rat_negLLall[count-1] = np.mean(sub_rat)

print('Human', np.mean(Human_negLLall*2))
print('Pigeon', np.mean(Pigeon_negLLall*2))
print('Rat', np.mean(Rat_negLLall*2))