#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Calculate correlation between parameters
*alpha-eps
*eps-lambda
*alpha-lambda
"""
import time
start = time.time()
from scipy import stats             # statistical tools
import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import matplotlib.pyplot as plt     # plotting
from sklearn.metrics import r2_score
import matplotlib

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'HPC_fit7models')
human_dir = os.path.join(data_dir, 'Human_output')
pigeon_dir = os.path.join(data_dir, 'Pigeon_output')
rat_dir = os.path.join(data_dir, 'Rat_output/average')


human_file = os.path.join(human_dir, 'Human_M2.csv')
pigeon_file = os.path.join(pigeon_dir, 'Pigeon_M2.csv')
rat_file = os.path.join(rat_dir, 'Ratav_M2.csv')

human_params = pd.read_csv(human_file)
pigeon_params = pd.read_csv(pigeon_file)
rat_params = pd.read_csv(rat_file)

human_eps = human_params['eps'].to_numpy()
human_alpha = human_params['alpha'].to_numpy()
human_lambda = human_params['lambda'].to_numpy()

pigeon_eps = pigeon_params['eps'].to_numpy()
pigeon_alpha = pigeon_params['alpha'].to_numpy()
pigeon_lambda = pigeon_params['lambda'].to_numpy()

rat_eps = rat_params['eps'].to_numpy()
rat_alpha = rat_params['alpha'].to_numpy()
rat_lambda = rat_params['lambda'].to_numpy()

all_eps = np.concatenate((human_eps, pigeon_eps, rat_eps))
all_alpha = np.concatenate((human_alpha, pigeon_alpha, rat_alpha))
all_lambda = np.concatenate((human_lambda, pigeon_lambda, rat_lambda))
data_labels = np.concatenate((['human'] * len(human_eps),
                              ['pigeon'] * len(pigeon_eps),
                              ['rat'] * len(rat_eps)))

#correlation calculations
#Pearson correlation coefficients
eps_alpha_pearson = stats.pearsonr(all_eps, all_alpha)
eps_lambda_pearson = stats.pearsonr(all_eps, all_lambda)
alpha_lambda_pearson = stats.pearsonr(all_alpha, all_lambda)

#Spearman correlation coefficients
eps_alpha_spearman = stats.spearmanr(all_eps, all_alpha)
eps_lambda_spearman = stats.spearmanr(all_eps, all_lambda)
alpha_lambda_spearman = stats.spearmanr(all_alpha, all_lambda)

#R2 scores values
eps_alpha_r2 = r2_score(all_eps, all_alpha)
eps_lambda_r2 = r2_score(all_eps, all_lambda)
alpha_lambda_r2 = r2_score(all_alpha, all_lambda)

eps_alpha_human_pearson = stats.pearsonr(human_eps, human_alpha)
eps_lambda_human_pearson = stats.pearsonr(human_eps, human_lambda)
alpha_lambda_human_pearson = stats.pearsonr(human_alpha, human_lambda)

eps_alpha_human_spearman = stats.spearmanr(human_eps, human_alpha)
eps_lambda_human_spearman = stats.spearmanr(human_eps, human_lambda)
alpha_lambda_human_spearman = stats.spearmanr(human_alpha, human_lambda)

eps_alpha_pigeon_pearson = stats.pearsonr(pigeon_eps, pigeon_alpha)
eps_lambda_pigeon_pearson = stats.pearsonr(pigeon_eps, pigeon_lambda)
alpha_lambda_pigeon_pearson = stats.pearsonr(pigeon_alpha, pigeon_lambda)

eps_alpha_pigeon_spearman = stats.spearmanr(pigeon_eps, pigeon_alpha)
eps_lambda_pigeon_spearman = stats.spearmanr(pigeon_eps, pigeon_lambda)
alpha_lambda_pigeon_spearman = stats.spearmanr(pigeon_alpha, pigeon_lambda)

eps_alpha_rat_pearson = stats.pearsonr(rat_eps, rat_alpha)
eps_lambda_rat_pearson = stats.pearsonr(rat_eps, rat_lambda)
alpha_lambda_rat_pearson = stats.pearsonr(rat_alpha, rat_lambda)

eps_alpha_rat_spearman = stats.spearmanr(rat_eps, rat_alpha)
eps_lambda_rat_spearman = stats.spearmanr(rat_eps, rat_lambda)
alpha_lambda_rat_spearman = stats.spearmanr(rat_alpha, rat_lambda)

print('Pearson eps-alpha human', eps_alpha_human_pearson)
print('Pearson eps-lambda human', eps_lambda_human_pearson)
print('Pearson alpha-lambda human', alpha_lambda_human_pearson)
print('Spearman eps-alpha human', eps_alpha_human_spearman)
print('Spearman eps-lambda human', eps_lambda_human_spearman)
print('Spearman alpha-lambda human', alpha_lambda_human_spearman)
print('                           ')
print('Pearson eps-alpha pigeon', eps_alpha_pigeon_pearson)
print('Pearson eps-lambda pigeon', eps_lambda_pigeon_pearson)
print('Pearson alpha-lambda pigeon', alpha_lambda_pigeon_pearson)
print('Spearman eps-alpha pigeon', eps_alpha_pigeon_spearman)
print('Spearman eps-lambda pigeon', eps_lambda_pigeon_spearman)
print('Spearman alpha-lambda pigeon', alpha_lambda_pigeon_spearman)
print('                           ')
print('Pearson eps-alpha rat', eps_alpha_rat_pearson)
print('Pearson eps-lambda rat', eps_lambda_rat_pearson)
print('Pearson alpha-lambda rat', alpha_lambda_rat_pearson)
print('Spearman eps-alpha rat', eps_alpha_rat_spearman)
print('Spearman eps-lambda rat', eps_lambda_rat_spearman)
print('Spearman alpha-lambda rat', alpha_lambda_rat_spearman)



matplotlib.rcParams['font.family'] = 'times new roman'

#Scatter plots
fontsize = 15
fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(18,6))
ax1.scatter(human_eps, human_alpha, c='#695a39', alpha=0.8)
ax1.scatter(pigeon_eps, pigeon_alpha, c="#3f87da", alpha=0.8)
ax1.scatter(rat_eps, rat_alpha, c="#698286", alpha=0.8)
#plt.title(f'Epsilon vs Alpha\nPearson: {eps_alpha_pearson[0]:.2f}, p {eps_alpha_pearson[1]:.2f}, Spearman: {eps_alpha_spearman[0]:.2f}, p {eps_alpha_spearman[1]:.2f}, R2: {eps_alpha_r2:.2f}')
ax1.set_xlabel(r'$\epsilon$', fontsize=fontsize)
ax1.set_ylabel(r'$\alpha$', fontsize=fontsize)
ax1.set_ylim([0,1])
ax1.set_xlim([0,1])

#plt.savefig(os.path.join(data_dir, 'paramcorr/epsalpha.png'), dpi=300, bbox_inches='tight')

ax2.scatter(human_eps, human_lambda, c='#695a39', alpha=0.8)
ax2.scatter(pigeon_eps, pigeon_lambda, c="#3f87da", alpha=0.8)
ax2.scatter(rat_eps, rat_lambda, c="#698286", alpha=0.8)
#plt.title(f'Epsilon vs Lambda\nPearson: {eps_lambda_pearson[0]:.2f}, p {eps_lambda_pearson[1]:.2f}, Spearman: {eps_lambda_spearman[0]:.2f}, p {eps_lambda_spearman[1]:.2f}, R2: {eps_lambda_r2:.2f}')
ax2.set_xlabel(r'$\epsilon$', fontsize=fontsize)
ax2.set_ylabel(r'$\lambda$', fontsize=fontsize)
ax2.set_xlim([0,1])
ax2.set_ylim([-0.5,0.5])
#plt.savefig(os.path.join(data_dir, 'paramcorr/epslambda.png'), dpi=300, bbox_inches='tight')

ax3.scatter(human_alpha, human_lambda, c='#695a39', alpha=0.8)
ax3.scatter(pigeon_alpha, pigeon_lambda, c="#3f87da", alpha=0.8)
ax3.scatter(rat_alpha, rat_lambda, c="#698286", alpha=0.8)
#plt.title(f'Alpha vs Lambda\nPearson: {alpha_lambda_pearson[0]:.2f}, p {alpha_lambda_pearson[1]:.2f}, Spearman: {alpha_lambda_spearman[0]:.2f}, p {alpha_lambda_spearman[1]:.2f}, R2: {alpha_lambda_r2:.2f}')
ax3.set_xlabel(r'$\alpha$', fontsize=fontsize)
ax3.set_ylabel(r'$\lambda$', fontsize=fontsize)
ax3.set_ylim([-0.5,0.5])
ax3.set_xlim([0,1])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
#plt.savefig(os.path.join(data_dir, 'paramcorr/alphalambda.png'), dpi=300, bbox_inches='tight')
plt.show()