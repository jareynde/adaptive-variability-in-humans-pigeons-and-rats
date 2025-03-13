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
Plot heatmaps used in paper
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
import statsmodels.stats.multitest as smm
import matplotlib


matplotlib.rcParams['font.family'] = 'times new roman'
matplotlib.rcParams['font.size'] = 75
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = "reward"
save_dir = os.path.join(script_dir, folder_name)

eps0_adv_file = 'adv_mr_eps0.xlsx'
eps0_adv_dir = os.path.join(save_dir, eps0_adv_file)
eps0_adv = pd.read_excel(eps0_adv_dir)
eps0_adv=eps0_adv.to_numpy()
eps0_adv=eps0_adv[::-1]
eps0_adv=eps0_adv[:,1:]
print(np.shape(eps0_adv))

eps24_adv_file = 'adv_mr_eps12.xlsx'
eps24_adv_dir = os.path.join(save_dir, eps24_adv_file)
eps24_adv = pd.read_excel(eps24_adv_dir)
eps24_adv=eps24_adv.to_numpy()
eps24_adv=eps24_adv[::-1]
eps24_adv=eps24_adv[:,1:]

eps72_adv_file = 'adv_mr_eps36.xlsx'
eps72_adv_dir = os.path.join(save_dir, eps72_adv_file)
eps72_adv = pd.read_excel(eps72_adv_dir)
eps72_adv=eps72_adv.to_numpy()
eps72_adv=eps72_adv[::-1]
eps72_adv=eps72_adv[:,1:]



eps0_sta_file = 'sta_mr_eps0.xlsx'
eps0_sta_dir = os.path.join(save_dir, eps0_sta_file)
eps0_sta = pd.read_excel(eps0_sta_dir)
eps0_sta=eps0_sta.to_numpy()
eps0_sta=eps0_sta[::-1]
eps0_sta=eps0_sta[:,1:]

eps24_sta_file = 'sta_mr_eps12.xlsx'
eps24_sta_dir = os.path.join(save_dir, eps24_sta_file)
eps24_sta = pd.read_excel(eps24_sta_dir)
eps24_sta=eps24_sta.to_numpy()
eps24_sta=eps24_sta[::-1]
eps24_sta=eps24_sta[:,1:]

eps72_sta_file = 'sta_mr_eps36.xlsx'
eps72_sta_dir = os.path.join(save_dir, eps72_sta_file)
eps72_sta = pd.read_excel(eps72_sta_dir)
eps72_sta=eps72_sta.to_numpy()
eps72_sta=eps72_sta[::-1]
eps72_sta=eps72_sta[:,1:]



eps0_vol_file = 'vol_mr_eps0.xlsx'
eps0_vol_dir = os.path.join(save_dir, eps0_vol_file)
eps0_vol = pd.read_excel(eps0_vol_dir)
eps0_vol=eps0_vol.to_numpy()
eps0_vol=eps0_vol[::-1]
eps0_vol=eps0_vol[:,1:]

eps24_vol_file = 'vol_mr_eps12.xlsx'
eps24_vol_dir = os.path.join(save_dir, eps24_vol_file)
eps24_vol = pd.read_excel(eps24_vol_dir)
eps24_vol=eps24_vol.to_numpy()
eps24_vol=eps24_vol[::-1]
eps24_vol=eps24_vol[:,1:]

eps72_vol_file = 'vol_mr_eps36.xlsx'
eps72_vol_dir = os.path.join(save_dir, eps72_vol_file)
eps72_vol = pd.read_excel(eps72_vol_dir)
eps72_vol=eps72_vol.to_numpy()
eps72_vol=eps72_vol[::-1]
eps72_vol=eps72_vol[:,1:]

fontsize=70
heat = [0,12.5,25,37.5,50]
all_eps = np.linspace(0, 1, 51, endpoint=True)
all_lr = [1, 0.75, 0.5, 0.25, 0]
all_unchosen = [-1, -0.5, 0, 0.5, 1]
cmappie = 'cividis'
title = os.path.join(save_dir, '1adveps0')
fig, ax = plt.subplots(figsize=(28, 28))
im = ax.imshow(eps0_adv, vmin=0, vmax=0.65, cmap=cmappie)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(fr"Adversarial environment" "\n" "epsilon ($\epsilon$) = 0")
plt.savefig(title)

title = os.path.join(save_dir, '1adveps24')
fig, ax = plt.subplots(figsize=(28, 28))
im = ax.imshow(eps24_adv, vmin=0, vmax=0.65, cmap=cmappie)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Adversarial environment" "\n" "epsilon ($\epsilon$) = 0.24")
plt.savefig(title)

title = os.path.join(save_dir, '1adveps72')
fig, ax = plt.subplots(figsize=(28, 28))
max_idx = np.unravel_index(np.argmax(eps72_adv), eps72_adv.shape)
im = ax.imshow(eps72_adv, vmin=0, vmax=0.65, cmap=cmappie)
ax.scatter(max_idx[1], max_idx[0], color='red', marker='.', s=300)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Adversarial environment" "\n" "epsilon ($\epsilon$) = 0.72")
plt.savefig(title)





title = os.path.join(save_dir, '1staeps0')
fig, ax = plt.subplots(figsize=(28, 28))
max_idx = np.unravel_index(np.argmax(eps0_sta), eps0_sta.shape)
im = ax.imshow(eps0_sta, vmin=0.2, vmax=0.71, cmap=cmappie)
ax.scatter(max_idx[1], max_idx[0], color='red', marker='.', s=200)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Stable environment" "\n" "epsilon ($\epsilon$) = 0")
plt.savefig(title)

title = os.path.join(save_dir, '1staeps24')
fig, ax = plt.subplots(figsize=(28, 28))
im = ax.imshow(eps24_sta, vmin=0.2, vmax=0.71, cmap=cmappie)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Stable environment" "\n" "epsilon ($\epsilon$) = 0.24")
plt.savefig(title)


title = os.path.join(save_dir, '1staeps72')
fig, ax = plt.subplots(figsize=(28, 28))
im = ax.imshow(eps72_sta, vmin=0.2, vmax=0.71, cmap=cmappie)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Stable environment" "\n" "epsilon ($\epsilon$) = 0.72")
plt.savefig(title)






title = os.path.join(save_dir, '1voleps0')
fig, ax = plt.subplots(figsize=(28, 28))
max_idx = np.unravel_index(np.argmax(eps0_vol), eps0_vol.shape)
im = ax.imshow(eps0_vol, vmin=0.35, vmax=0.71, cmap=cmappie)
ax.scatter(max_idx[1], max_idx[0], color='red', marker='.', s=200)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Volatile environment" "\n" "epsilon ($\epsilon$) = 0")
plt.savefig(title)

title = os.path.join(save_dir, '1voleps24')
fig, ax = plt.subplots(figsize=(28, 28))
im = ax.imshow(eps24_vol, vmin=0.35, vmax=0.71, cmap=cmappie)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Volatile environment" "\n" "epsilon ($\epsilon$) = 0.24")
plt.savefig(title)

title = os.path.join(save_dir, '1voleps72')
fig, ax = plt.subplots(figsize=(28, 28))
im = ax.imshow(eps72_vol, vmin=0.35, vmax=0.71, cmap=cmappie)
ax.set_xlabel(r'Unchosen value-bias ($\lambda$)', fontsize=fontsize)
ax.set_ylabel(r'Learning rate ($\alpha$)', fontsize=fontsize)
ax.set_xticks(ticks=list(heat), labels=np.round(all_unchosen,1))  
ax.set_yticks(ticks=list(heat), labels=all_lr)  
cbar = plt.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
cbar.set_label('Average reward')
plt.title(r"Volatile environment" "\n" "epsilon ($\epsilon$) = 0.72")
plt.savefig(title)




#Figure 1: REWARDS GROUPED PER EPSILON VALUE
#plot rewards for each situation
#first bars are grouped per epsilon value
groups = [fr'$\epsilon$', fr'$\alpha$', fr'$\lambda$']
categories = ['stable environment', 'volatile environment', 'adversarial environment']
values = np.array([[0, 0, 0.72],
                   [0.18, 0.98, 0.16],
                   [-0.84, 0.04, 0.64]])


category_colors = ['darkcyan', 'darkorange', 'forestgreen']
group_colors = ['purple','red','gold']
fig, ax = plt.subplots()

bar_width = 0.2
x = np.arange(len(groups))

for j, category in enumerate(categories):
    category_values = values[:, j]
    
    # Plot the bars for the category
    for i in range(len(groups)):
        ax.bar(x[i] + j * bar_width, category_values[i], bar_width, capsize=5, color=category_colors[j], edgecolor = group_colors[i], linewidth=1.3, alpha=0.5, label=f'{category}' if i == 0 else "")

# Customize the plot
ax.set_ylabel('Parameter value')
plt.ylim([-1,1])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(groups)
#ax.legend()




plt.show()