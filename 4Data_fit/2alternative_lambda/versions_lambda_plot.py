#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Human data in seperate files per subject and condition
Model one: eps, alpha, lambda
Model two: eps, alpha
One epsilon, alpha and lambda_unchosen are estimated via a RW model

In this copy, we try multiple alternatives to lambda 
*1: lambda with stricter boundaries (1/K) (as with 1ML_lambda copy.py)
*2: lambda seperate from alpha RW and with weight in decision-policy
*3: lambda seperate from alpha RW but without weight
*4: lambda seperate from alpha RW with its own RW + weight
*5: binary lambda with weight (adding 1/K)
*6: binary lambda without weight (adding 1/K)
*7: binary lambda with weight (subtracting 1/K)
*8: binary lambda without weight (subtracting 1/K)

update after meeting on 2/7/2025:
*9: model 2 but weight is V = Q + wF and w unbounded
*10: model 4 but weight is V = Q + wF and w unbounded
*11: model 5 but weight is V = Q + wF and w unbounded
*12: model 7 but weight is V = Q + wF and w unbounded
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
import matplotlib


script_dir = os.path.dirname(os.path.abspath(__file__))
human_dir = os.path.join(script_dir, 'Human_output_versions_lambda/Human_BIC.xlsx')
pigeon_dir = os.path.join(script_dir, 'Pigeon_output_versions_lambda/Pigeon_BIC.xlsx')
rat_dir = os.path.join(script_dir, 'Rat_output_versions_lambda/Rats_av_BIC.xlsx')

human_bic = pd.read_excel(human_dir)
human_bic = human_bic.to_numpy()
human_bic = human_bic[:,1:]
print(human_bic.shape)

pigeon_bic = pd.read_excel(pigeon_dir)
pigeon_bic = pigeon_bic.to_numpy()
pigeon_bic = pigeon_bic[:,1:]
print(pigeon_bic.shape)

rat_bic = pd.read_excel(rat_dir)
rat_bic = rat_bic.to_numpy()
rat_bic = rat_bic[:,1:]
print(rat_bic.shape)


average_human = np.mean(human_bic, axis=0)
average_human = np.round(average_human, 2)
ste_human=np.std(human_bic, axis=0)/np.sqrt(18)
average_pigeon = np.mean(pigeon_bic, axis=0)
average_pigeon = np.round(average_pigeon,2)
ste_pigeon=np.std(pigeon_bic, axis=0)/np.sqrt(15)
average_rat = np.mean(rat_bic, axis=0)
average_rat = np.round(average_rat,2)
ste_rat=np.std(rat_bic, axis=0)/np.sqrt(20)


matplotlib.rcParams['font.family'] = 'times new roman'

models = (r"M $\lambda$1", r"M $\lambda$2", r"M $\lambda$3", r"M $\lambda$4.1", r"M $\lambda$4.2")
species_means = {
    'Human': (average_human[0], average_human[8], average_human[9], average_human[10], average_human[11]),
    'Pigeon': (average_pigeon[0], average_pigeon[8], average_pigeon[9], average_pigeon[10], average_pigeon[11]),
    'Rat': (average_rat[0], average_rat[8], average_rat[9], average_rat[10], average_rat[11]),
}
species_ste = {
    'Human': (ste_human[0], ste_human[8], ste_human[9], ste_human[10], ste_human[11]),
    'Pigeon': (ste_pigeon[0], ste_pigeon[8], ste_pigeon[9], ste_pigeon[10], ste_pigeon[11]),
    'Rat': (ste_rat[0], ste_rat[8], ste_rat[9], ste_rat[10], ste_rat[11]),
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in species_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('average BIC')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=3)
#ax.set_ylim(0, 250)

#plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.bar(models, species_means['Human'], color='#695a39')
ax1.set_ylim(47250, 47800)
ax1.set_ylabel('average BIC')
ax1.set_xlabel('Human')

ax2.bar(models, species_means['Pigeon'], color="#3f87da")
ax2.set_ylim(33040, 33300)
ax2.set_ylabel('average BIC')
ax2.set_xlabel('Pigeon')

ax3.bar(models, species_means['Rat'], color="#698286")
ax3.set_ylim(2870, 2940)
ax3.set_ylabel('average BIC')
ax3.set_xlabel('Rat')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.show()