#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import matplotlib.pyplot as plt     # plotting
import matplotlib
import scipy.stats as stats


data_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(data_dir, 'results/rat_results_eps_lr_lambda.xlsx')
data = pd.read_excel(file_dir)
true_epsilon = data['true_epsilon']
recov_epsilon = data['recov_epsilon']
true_alpha = data['true_alpha']
recov_alpha = data['recov_alpha']
true_unchosen = data['true_unchosen']
recov_unchosen = data['recov_unchosen']

fontsize = 20
matplotlib.rcParams['font.family'] = 'times new roman'


fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(18,6))
ax1.scatter(x=true_epsilon, y=recov_epsilon,s=2,c='black')
ax1.set_xlabel(r'true $\epsilon$ - stochasticity', fontsize=fontsize)
ax1.set_ylabel(r'recovered $\epsilon$ - stochasticity',fontsize=fontsize)


ax2.scatter(x=true_alpha, y=recov_alpha,s=2,c='black')
ax2.set_xlabel(r'true $\alpha$ - learning rate',fontsize=fontsize)
ax2.set_ylabel(r'recovered $\alpha$ - learning rate',fontsize=fontsize)

ax3.scatter(x=true_unchosen, y=recov_unchosen,s=2,c='black')
ax3.set_xlabel(r'true $\lambda$ - unchosen value-bias',fontsize=fontsize)
ax3.set_ylabel(r'recovered $\lambda$ - unchosen value-bias',fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=fontsize)
ax2.tick_params(axis='both', labelsize=fontsize)
ax3.tick_params(axis='both', labelsize=fontsize)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.show()


#Parameter recovery
def calculate_Rsquared(y_true, y_pred):
    # Calculate the mean of observed values
    mean_y_true = np.mean(y_true)

    # Calculate the Sum of Squared Residuals (SSR)
    ssr = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))

    # Calculate the Total Sum of Squares (SST)
    sst = sum((y_true[i] - mean_y_true)**2 for i in range(len(y_true)))

    # Calculate R-squared
    Rsquared = 1 - (ssr / sst)
    return Rsquared 

#correlations between estimated parameters
Reps_lr_2 = calculate_Rsquared(recov_epsilon, recov_alpha)
Reps_uc_2 = calculate_Rsquared(recov_epsilon, recov_unchosen)
Rlr_uc_2 = calculate_Rsquared(recov_alpha, recov_unchosen)

Peps_lr_2 = stats.pearsonr(recov_epsilon, recov_alpha)
Peps_uc_2 = stats.pearsonr(recov_epsilon, recov_unchosen)
Plr_uc_2 = stats.pearsonr(recov_alpha, recov_unchosen)

Seps_lr_2 = stats.spearmanr(recov_epsilon, recov_alpha)
Seps_uc_2 = stats.spearmanr(recov_epsilon, recov_unchosen)
Slr_uc_2 = stats.spearmanr(recov_alpha, recov_unchosen)



#plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model 2: correlations between recovered parameters', fontsize=16)
ax1.scatter(recov_epsilon, recov_alpha, color='black')
#ax1.plot([0, 1], [0, 1], color='black', linestyle='--')
ax1.set_title(f'Model 2: R²={Reps_lr_2:.2f}, \nPearson r={Peps_lr_2[0]:.2f} (p={Peps_lr_2[1]:.3f}), \nSpearman ρ={Seps_lr_2.correlation:.2f} (p={Seps_lr_2.pvalue:.3f})')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel(r'$\alpha$')
ax2.scatter(recov_epsilon, recov_unchosen, color='black')
#ax2.plot([0, 1], [0, 1], color='black', linestyle='--')
ax2.set_title(f'Model 2: R²={Reps_uc_2:.2f}, \nPearson r={Peps_uc_2[0]:.2f} (p={Peps_uc_2[1]:.3f}), \nSpearman ρ={Seps_uc_2.correlation:.2f} (p={Seps_uc_2.pvalue:.3f})')
ax2.set_xlabel(r'$\epsilon$')
ax2.set_ylabel(r'$\lambda$')
ax3.scatter(recov_alpha, recov_unchosen, color=('black'))
#ax3.plot([-1/K, 1/K], [-1/K, 1/K], color='black', linestyle='--')
ax3.set_title(f'Model 2: R²={Rlr_uc_2:.2f}, \nPearson r={Plr_uc_2[0]:.2f} (p={Plr_uc_2[1]:.3f}), \nSpearman ρ={Slr_uc_2.correlation:.2f} (p={Slr_uc_2.pvalue:.3f})')
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel(r'$\lambda$')  
plt.show()