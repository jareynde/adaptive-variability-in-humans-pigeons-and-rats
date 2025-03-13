#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Model recovery
Model one: eps, alpha, lambda
Model two: eps, alpha, rho
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

random.seed(19287)
np.random.seed(290)

save_dir = os.path.dirname(os.path.abspath(__file__))


#this function returns an array with the amount of times each model 'won', i.e., had the lowest BIC
#on index 0, amount of time the context-specific eps model won, and so on:
#0 eps-alpha-lambda
#1 eps-alpha-rho
#2 eps-alpha
#3 0eps-alpha-lambda 
def count(n):
    # Initialize a list to store the counts for each integer from 0 to 1
    counts = [0] * 2
    
    # Iterate through the array 'n' and count occurrences
    for num in n:
        if 0 <= num <= 1:
            counts[num] += 1

    return counts



### RATS ###
BIC1_data = os.path.join(save_dir, '1rat_BIC_eps-alpha-lambda.xlsx' )
BIC1 = pd.read_excel(BIC1_data)
BIC1  =BIC1.to_numpy()
BIC1 = BIC1[:,1:]

BIC2_data = os.path.join(save_dir, '2rat_BIC_eps-alpha.xlsx' )
BIC2 = pd.read_excel(BIC2_data)
BIC2 = BIC2.to_numpy()
BIC2 = BIC2[:,1:]



min_BIC1 = np.argmin(BIC1, axis=1)
min_BIC2 = np.argmin(BIC2, axis=1)


amount=len(min_BIC1)


cBIC1 = count(min_BIC1)
cBIC2 = count(min_BIC2)

BIC = np.array([cBIC1, cBIC2])/amount
print(cBIC1)
print(BIC)

#y axis: data simulated from
#x axis: best model fit

simulations = [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']
model_fits =  [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']

fig, ax = plt.subplots()
im = ax.imshow(BIC, cmap='Greens')
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(model_fits)), labels=model_fits, fontsize=15)
ax.set_yticks(np.arange(len(simulations)), labels=simulations, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(simulations)):
    for j in range(len(model_fits)):
        text = ax.text(j, i, BIC[i, j],
                       ha="center", va="center", color="dimgrey")

ax.set_title("RAT confusion matrix - p(fit model|simulated model)",fontsize=15)
ax.set_xlabel('% best model fit')
ax.set_ylabel('data simulated from')
fig.colorbar(im, ax=ax)
plt.show()





### HP8 ###
BIC1_data = os.path.join(save_dir, '1HP8_BIC_eps-alpha-lambda.xlsx' )
BIC1 = pd.read_excel(BIC1_data)
BIC1  =BIC1.to_numpy()
BIC1 = BIC1[:,1:]

BIC2_data = os.path.join(save_dir, '2HP8_BIC_eps-alpha.xlsx' )
BIC2 = pd.read_excel(BIC2_data)
BIC2 = BIC2.to_numpy()
BIC2 = BIC2[:,1:]



min_BIC1 = np.argmin(BIC1, axis=1)
min_BIC2 = np.argmin(BIC2, axis=1)


amount=len(min_BIC1)


cBIC1 = count(min_BIC1)
cBIC2 = count(min_BIC2)

BIC = np.array([cBIC1, cBIC2])/amount
print(cBIC1)
print(BIC)

#y axis: data simulated from
#x axis: best model fit

simulations = [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']
model_fits =  [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']

fig, ax = plt.subplots()
im = ax.imshow(BIC, cmap='Greens')
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(model_fits)), labels=model_fits, fontsize=15)
ax.set_yticks(np.arange(len(simulations)), labels=simulations, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(simulations)):
    for j in range(len(model_fits)):
        text = ax.text(j, i, BIC[i, j],
                       ha="center", va="center", color="dimgrey")

ax.set_title("HP8 confusion matrix - p(fit model|simulated model)",fontsize=15)
ax.set_xlabel('% best model fit')
ax.set_ylabel('data simulated from')
fig.colorbar(im, ax=ax)
plt.show()






### HP4 ###
BIC1_data = os.path.join(save_dir, '1HP4_BIC_eps-alpha-lambda.xlsx' )
BIC1 = pd.read_excel(BIC1_data)
BIC1  =BIC1.to_numpy()
BIC1 = BIC1[:,1:]

BIC2_data = os.path.join(save_dir, '2HP4_BIC_eps-alpha.xlsx' )
BIC2 = pd.read_excel(BIC2_data)
BIC2 = BIC2.to_numpy()
BIC2 = BIC2[:,1:]



min_BIC1 = np.argmin(BIC1, axis=1)
min_BIC2 = np.argmin(BIC2, axis=1)


amount=len(min_BIC1)


cBIC1 = count(min_BIC1)
cBIC2 = count(min_BIC2)

BIC = np.array([cBIC1, cBIC2])/amount
print(cBIC1)
print(BIC)

#y axis: data simulated from
#x axis: best model fit

simulations = [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']
model_fits =  [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']

fig, ax = plt.subplots()
im = ax.imshow(BIC, cmap='Greens')
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(model_fits)), labels=model_fits, fontsize=15)
ax.set_yticks(np.arange(len(simulations)), labels=simulations, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(simulations)):
    for j in range(len(model_fits)):
        text = ax.text(j, i, BIC[i, j],
                       ha="center", va="center", color="dimgrey")

ax.set_title("HP4 confusion matrix - p(fit model|simulated model)",fontsize=15)
ax.set_xlabel('% best model fit')
ax.set_ylabel('data simulated from')
fig.colorbar(im, ax=ax)
plt.show()




### HP2 ###
BIC1_data = os.path.join(save_dir, '1HP2_BIC_eps-alpha-lambda.xlsx' )
BIC1 = pd.read_excel(BIC1_data)
BIC1  =BIC1.to_numpy()
BIC1 = BIC1 [:,1:]

BIC2_data = os.path.join(save_dir, '2HP2_BIC_eps-alpha.xlsx' )
BIC2 = pd.read_excel(BIC2_data)
BIC2 = BIC2.to_numpy()
BIC2 = BIC2[:,1:]



min_BIC1 = np.argmin(BIC1, axis=1)
min_BIC2 = np.argmin(BIC2, axis=1)


amount=len(min_BIC1)


cBIC1 = count(min_BIC1)
cBIC2 = count(min_BIC2)

BIC = np.array([cBIC1, cBIC2])/amount
print(cBIC1)
print(BIC)

#y axis: data simulated from
#x axis: best model fit

simulations = [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']
model_fits =  [r'$\epsilon$-$\alpha$-$\lambda$',
               r'$\epsilon$-$\alpha$']

fig, ax = plt.subplots()
im = ax.imshow(BIC, cmap='Greens')
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(model_fits)), labels=model_fits, fontsize=15)
ax.set_yticks(np.arange(len(simulations)), labels=simulations, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(simulations)):
    for j in range(len(model_fits)):
        text = ax.text(j, i, BIC[i, j],
                       ha="center", va="center", color="dimgrey")

ax.set_title("HP2 confusion matrix - p(fit model|simulated model)",fontsize=15)
ax.set_xlabel('% best model fit')
ax.set_ylabel('data simulated from')
fig.colorbar(im, ax=ax)
plt.show()