#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
Fitting of the Neuringer data for humans
Human data in seperate files per subject and condition

Plot 6 models fit to data
*1: eps-lr
*2: eps-lr-uc -> wins for rat
*3: eps-lr-gam

*4: eps-lr-uc-gam -> wins for pigeon
*5: eps-lr-uc-ucs/w

*7: eps-lr-uc-ucs/w-gam -> wins for human
"""

import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import matplotlib.pyplot as plt     # plotting functions
import matplotlib
import ptitprince as pt
import seaborn as sns

matplotlib.rcParams['font.family'] = 'times new roman'

script_dir = os.path.dirname(os.path.abspath(__file__))
human_dir = os.path.join(script_dir, 'Human_output')
human_files = os.listdir(human_dir)
pigeon_dir = os.path.join(script_dir, 'Pigeon_output')
pigeon_files = os.listdir(pigeon_dir)
rat_dir = os.path.join(script_dir, 'Rat_output/average')
rat_files = os.listdir(rat_dir)

model_labels = ['M1', 'M2', 'M3.1', 'M3.2', 'M4.1', 'M4.2', 'M5', 'M7.1', 'M7.2']
human_bic = pd.DataFrame()
for count, file in enumerate(human_files):
    file_dir = os.path.join(human_dir, file)
    df = pd.read_csv(file_dir)
    human_bic[f'{model_labels[count]}'] = df['BIC']

    human_bic_np = human_bic.to_numpy()

human_bic_arg = np.argmin(human_bic_np, axis=1)
human_bic_count = np.zeros(len(model_labels))
for i in range(len(model_labels)):
    human_bic_count[i] = np.shape(np.where(human_bic_arg == i)[0])[0]
    
human_bic_mean = np.mean(human_bic_np, axis=0)





pigeon_bic = pd.DataFrame()
for count, file in enumerate(pigeon_files):
    file_dir = os.path.join(pigeon_dir, file)
    df = pd.read_csv(file_dir)
    pigeon_bic[f'{model_labels[count]}'] = df['BIC']

    pigeon_bic_np = pigeon_bic.to_numpy()

pigeon_bic_arg = np.argmin(pigeon_bic_np, axis=1)
pigeon_bic_count = np.zeros(len(model_labels))
for i in range(len(model_labels)):
    pigeon_bic_count[i] = np.shape(np.where(pigeon_bic_arg == i)[0])[0]
    
pigeon_bic_mean = np.mean(pigeon_bic_np, axis=0)






rat_bic = pd.DataFrame()
for count, file in enumerate(rat_files):
    file_dir = os.path.join(rat_dir, f'{file}')
    df = pd.read_csv(file_dir)
    rat_bic[f'{model_labels[count]}'] = df['BIC']

    rat_bic_np = rat_bic.to_numpy()

rat_bic_arg = np.argmin(rat_bic_np, axis=1)
rat_bic_count = np.zeros(len(model_labels))
for i in range(len(model_labels)):
    rat_bic_count[i] = np.shape(np.where(rat_bic_arg == i)[0])[0]

    
rat_bic_mean = np.mean(rat_bic_np, axis=0)


'''
fontsize = 20
model_labels = ['M1', 'M2.1', 'M2.2', 'M3.1', 'M3.2', 'M4', 'M5.1', 'M5.2']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
#put on scientific notation of y-axis
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax1.bar(model_labels, human_bic_mean[1:], color='#695a39', edgecolor='black')
ax1.set_ylim(46600, 47800)
ax2.bar(model_labels, pigeon_bic_mean[1:], color="#3f87da", edgecolor='black')
ax2.set_ylim(32800,33350)
ax3.bar(model_labels, rat_bic_mean[1:], color="#698286", edgecolor='black')
ax3.set_ylim(2860,2950)
ax1.set_xticklabels(model_labels, rotation=45, fontsize=fontsize)
ax2.set_xticklabels(model_labels, rotation=45, fontsize=fontsize)
ax3.set_xticklabels(model_labels, rotation=45, fontsize=fontsize)
ax1.set_ylabel('mean BIC', fontsize=fontsize)
ax2.set_ylabel('mean BIC', fontsize=fontsize)
ax3.set_ylabel('mean BIC', fontsize=fontsize)
ax1.set_title('Human', fontsize=fontsize)
ax2.set_title('Pigeon', fontsize=fontsize)
ax3.set_title('Rat', fontsize=fontsize)
fig.suptitle('BIC means per model')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax1.tick_params(axis='y', labelsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.plot()


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
ax1.bar(model_labels, human_bic_count[1:], color='#695a39')
ax2.bar(model_labels, pigeon_bic_count[1:], color="#3f87da")
ax3.bar(model_labels, rat_bic_count[1:], color="#698286")
ax1.set_xticklabels(model_labels, rotation=45)
ax2.set_xticklabels(model_labels, rotation=45)
ax3.set_xticklabels(model_labels, rotation=45)
ax1.set_ylabel('number of participants')
ax2.set_ylabel('number of participants')
ax3.set_ylabel('number of participants')
ax1.set_title('Human')
ax2.set_title('Pigeon')
ax3.set_title('Rat')
fig.suptitle('Winning model counts')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
plt.plot()

#plt.show()
'''

###########################################################################
#plotting parameters of winning models
############################################################################

human_file = os.path.join(human_dir, 'Human_M7_1.csv')
pigeon_file = os.path.join(pigeon_dir, 'Pigeon_M4_1.csv')
rat_file = os.path.join(rat_dir, 'Ratav_M2.csv')

human_data = pd.read_csv(human_file)
pigeon_data = pd.read_csv(pigeon_file)
rat_data = pd.read_csv(rat_file)


all_labels =[]
all_labels.append(human_data['label'])
all_labels.append(pigeon_data['label'])
all_labels.append(rat_data['label'])

all_choices = []
all_choices.append(human_data['choices'])
all_choices.append(pigeon_data['choices']+10)
all_choices.append(rat_data['choices'])

epsilon=[]
epsilon.append(human_data['eps'])
epsilon.append(pigeon_data['eps'])
epsilon.append(rat_data['eps'])

alpha = []
alpha.append(human_data['alpha'])
alpha.append(pigeon_data['alpha'])
alpha.append(rat_data['alpha'])

lambdas = []
lambdas.append(human_data['lambda']*human_data['choices'])
lambdas.append(pigeon_data['lambda']*pigeon_data['choices'])
lambdas.append(rat_data['lambda']*rat_data['choices'])

lambdaseq = []
lambdaseq.append(human_data['lambda_s'])
lambdaseq.append(np.zeros(pigeon_data['lambda'].shape))
lambdaseq.append(np.zeros(rat_data['lambda'].shape))

weight = []
weight.append(human_data['w'])
weight.append(np.ones(pigeon_data['lambda'].shape))
weight.append(np.ones(rat_data['lambda'].shape))

gamma = []
gamma.append(human_data['gamr'])
gamma.append(pigeon_data['gamr'])
gamma.append(0.5*np.ones(rat_data['lambda'].shape))






figsize=(7,2.5)
purple = (0.72, 0.54, 0.86)
red = (1, 0.58, 0.58)
yellow = (1, 0.86, 0.41)
orange = (255/255, 174/255, 67/255)
orange2 = (204/255, 102/255, 0/255)
turquoise = (138/255, 222/255, 184/255)
custom_palette = {2: "#493710", 4: "#88661E", 8: "#ddae48", 12: "#08203B", 14: "#285385", 18: "#3f87da" , 5: '#698286', 500: 'black'} 
fontsize = 12

# Combine features in a list
data_x =  epsilon


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [purple, purple, purple]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [purple, purple, purple]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)


# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)


# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\epsilon$ - stochasticity', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/winning_models/epsilon.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  alpha


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [red, red, red]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [red, red, red]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\alpha$ - learning rate', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/winning_models/alpha.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  lambdas


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [yellow, yellow, yellow]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [yellow, yellow, yellow]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$ - unchosen value-bias', fontsize=fontsize)
plt.ylim([-1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/winning_models/lambda.png'), dpi=300, bbox_inches='tight')




# Combine features in a list
data_x =  gamma

print(data_x)
# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [turquoise, turquoise, turquoise]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [turquoise, turquoise, turquoise]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)


# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)


# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)


# Add labels and legend
plt.ylabel(r'$\gamma$$_r$ - episodic memory', fontsize=fontsize)
plt.ylim([-0.1,1]) #this line totally messes with the correct values, I don't know why?
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)

plt.savefig(os.path.join(script_dir, 'save_plots/winning_models/gamma_recent.png'), dpi=300, bbox_inches='tight')









# Combine features in a list
data_x =  lambdaseq


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [orange, orange, orange]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [orange, orange, orange]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$$_{seq}$ - unchosen sequence-bias', fontsize=fontsize)
plt.ylim([-1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/winning_models/lambda_seq.png'), dpi=300, bbox_inches='tight')





# Combine features in a list
data_x =  weight


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [orange2, orange2, orange2]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [orange2, orange2, orange2]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\omega$ - weighting factor', fontsize=fontsize)
plt.ylim([0,10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/winning_models/weight.png'), dpi=300, bbox_inches='tight')

plt.show()






###########################################################################
#plotting parameters of M2
############################################################################

human_file = os.path.join(human_dir, 'Human_M2.csv')
pigeon_file = os.path.join(pigeon_dir, 'Pigeon_M2.csv')
rat_file = os.path.join(rat_dir, 'Ratav_M2.csv')

human_data = pd.read_csv(human_file)
pigeon_data = pd.read_csv(pigeon_file)
rat_data = pd.read_csv(rat_file)


all_choices = []
all_choices.append(human_data['choices'])
all_choices.append(pigeon_data['choices']+10)
all_choices.append(rat_data['choices'])

epsilon=[]
epsilon.append(human_data['eps'])
epsilon.append(pigeon_data['eps'])
epsilon.append(rat_data['eps'])

alpha = []
alpha.append(human_data['alpha'])
alpha.append(pigeon_data['alpha'])
alpha.append(rat_data['alpha'])

lambdas = []
lambdas.append(human_data['lambda']*human_data['choices'])
lambdas.append(pigeon_data['lambda']*pigeon_data['choices'])
lambdas.append(rat_data['lambda']*rat_data['choices'])



figsize=(7,2.5)
purple = (0.72, 0.54, 0.86)
red = (1, 0.58, 0.58)
yellow = (1, 0.86, 0.41)
orange = (255/255, 174/255, 67/255)
orange2 = (204/255, 102/255, 0/255)
turquoise = (138/255, 222/255, 184/255)
custom_palette = {2: "#493710", 4: "#88661E", 8: "#ddae48", 12: "#08203B", 14: "#285385", 18: "#3f87da" , 5: '#698286', 500: 'black'} 
fontsize = 12

# Combine features in a list
data_x =  epsilon


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [purple, purple, purple]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [purple, purple, purple]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\epsilon$ - stochasticity', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M2/epsilon.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  alpha


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [red, red, red]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [red, red, red]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\alpha$ - learning rate', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M2/alpha.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  lambdas


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [yellow, yellow, yellow]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [yellow, yellow, yellow]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$ - unchosen value-bias', fontsize=fontsize)
plt.ylim([-1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M2/lambda.png'), dpi=300, bbox_inches='tight')


plt.show()








'''
###########################################################################
#plotting parameters of M7.2
############################################################################

human_file = os.path.join(human_dir, 'Human_M7_2.csv')
pigeon_file = os.path.join(pigeon_dir, 'Pigeon_M7_2.csv')
rat_file = os.path.join(rat_dir, 'Ratav_M7_2.csv')

human_data = pd.read_csv(human_file)
pigeon_data = pd.read_csv(pigeon_file)
rat_data = pd.read_csv(rat_file)


all_choices = []
all_choices.append(human_data['choices'])
all_choices.append(pigeon_data['choices']+10)
all_choices.append(rat_data['choices'])

epsilon=[]
epsilon.append(human_data['eps'])
epsilon.append(pigeon_data['eps'])
epsilon.append(rat_data['eps'])

alpha = []
alpha.append(human_data['alpha'])
alpha.append(pigeon_data['alpha'])
alpha.append(rat_data['alpha'])

lambdas = []
lambdas.append(human_data['lambda']*human_data['choices'])
lambdas.append(pigeon_data['lambda']*pigeon_data['choices'])
lambdas.append(rat_data['lambda']*rat_data['choices'])

lambdaseq = []
lambdaseq.append(human_data['lambda_s'])
lambdaseq.append(pigeon_data['lambda_s'])
lambdaseq.append(rat_data['lambda_s'])

weight = []
weight.append(human_data['w'])
weight.append(pigeon_data['w'])
weight.append(rat_data['w'])

gamma = []
gamma.append(human_data['gamnr'])
gamma.append(pigeon_data['gamnr'])
gamma.append(rat_data['gamnr'])







figsize=(7,2.5)
purple = (0.72, 0.54, 0.86)
red = (1, 0.58, 0.58)
yellow = (1, 0.86, 0.41)
orange = (255/255, 174/255, 67/255)
orange2 = (204/255, 102/255, 0/255)
turquoise = (138/255, 222/255, 184/255)
custom_palette = {2: "#493710", 4: "#88661E", 8: "#ddae48", 12: "#08203B", 14: "#285385", 18: "#3f87da" , 5: '#698286', 500: 'black'} 
fontsize = 12

# Combine features in a list
data_x =  epsilon


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [purple, purple, purple]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [purple, purple, purple]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\epsilon$ - stochasticity', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_2/epsilon.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  alpha


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [red, red, red]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [red, red, red]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\alpha$ - learning rate', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_2/alpha.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  lambdas


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [yellow, yellow, yellow]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [yellow, yellow, yellow]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$ - unchosen value-bias', fontsize=fontsize)
plt.ylim([-0.5,0.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_2/lambda.png'), dpi=300, bbox_inches='tight')





# Combine features in a list
data_x =  gamma


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [turquoise, turquoise, turquoise]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [turquoise, turquoise, turquoise]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\gamma$$_{nr}$ - episodic memory', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_2/gamma_nonrecent.png'), dpi=300, bbox_inches='tight')








# Combine features in a list
data_x =  lambdaseq


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [orange, orange, orange]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [orange, orange, orange]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$$_{seq}$ - unchosen sequence-bias', fontsize=fontsize)
plt.ylim([-0.5,0.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_2/lambda_seq.png'), dpi=300, bbox_inches='tight')





# Combine features in a list
data_x =  weight


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [orange2, orange2, orange2]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [orange2, orange2, orange2]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\omega$ - weighting factor', fontsize=fontsize)
plt.ylim([0,10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_2/weight.png'), dpi=300, bbox_inches='tight')

plt.show()










###########################################################################
#plotting parameters of model 7.1
############################################################################

human_file = os.path.join(human_dir, 'Human_M7_1.csv')
pigeon_file = os.path.join(pigeon_dir, 'Pigeon_M7_1.csv')
rat_file = os.path.join(rat_dir, 'Ratav_M7_1.csv')

human_data = pd.read_csv(human_file)
pigeon_data = pd.read_csv(pigeon_file)
rat_data = pd.read_csv(rat_file)



all_choices = []
all_choices.append(human_data['choices'])
all_choices.append(pigeon_data['choices']+10)
all_choices.append(rat_data['choices'])

epsilon=[]
epsilon.append(human_data['eps'])
epsilon.append(pigeon_data['eps'])
epsilon.append(rat_data['eps'])

alpha = []
alpha.append(human_data['alpha'])
alpha.append(pigeon_data['alpha'])
alpha.append(rat_data['alpha'])

lambdas = []
lambdas.append(human_data['lambda']*human_data['choices'])
lambdas.append(pigeon_data['lambda']*pigeon_data['choices'])
lambdas.append(rat_data['lambda']*rat_data['choices'])

lambdaseq = []
lambdaseq.append(human_data['lambda_s'])
lambdaseq.append(pigeon_data['lambda_s'])
lambdaseq.append(rat_data['lambda_s'])

weight = []
weight.append(human_data['w'])
weight.append(pigeon_data['w'])
weight.append(rat_data['w'])

gamma = []
gamma.append(human_data['gamr'])
gamma.append(pigeon_data['gamr'])
gamma.append(rat_data['gamr'])







figsize=(7,2.5)
purple = (0.72, 0.54, 0.86)
red = (1, 0.58, 0.58)
yellow = (1, 0.86, 0.41)
orange = (255/255, 174/255, 67/255)
orange2 = (204/255, 102/255, 0/255)
turquoise = (138/255, 222/255, 184/255)
custom_palette = {2: "#493710", 4: "#88661E", 8: "#ddae48", 12: "#08203B", 14: "#285385", 18: "#3f87da" , 5: '#698286', 500: 'black'} 
fontsize = 12

# Combine features in a list
data_x =  epsilon


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [purple, purple, purple]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [purple, purple, purple]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\epsilon$ - stochasticity', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_1/epsilon.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  alpha


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [red, red, red]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [red, red, red]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\alpha$ - learning rate', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_1/alpha.png'), dpi=300, bbox_inches='tight')






# Combine features in a list
data_x =  lambdas


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [yellow, yellow, yellow]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [yellow, yellow, yellow]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$ - unchosen value-bias', fontsize=fontsize)
plt.ylim([-0.5,0.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_1/lambda.png'), dpi=300, bbox_inches='tight')





# Combine features in a list
data_x =  gamma

# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [turquoise, turquoise, turquoise]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [turquoise, turquoise, turquoise]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\gamma$$_r$ - episodic memory', fontsize=fontsize)
plt.ylim([-0.1,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_1/gamma_recent.png'), dpi=300, bbox_inches='tight')








# Combine features in a list
data_x =  lambdaseq


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [orange, orange, orange]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [orange, orange, orange]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\lambda$$_{seq}$ - unchosen sequence-bias', fontsize=fontsize)
plt.ylim([-0.5,0.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_1/lambda_seq.png'), dpi=300, bbox_inches='tight')





# Combine features in a list
data_x =  weight


# Start plotting
fig, ax = plt.subplots(figsize=figsize)

# Boxplot data
boxplots_colors = [orange2, orange2, orange2]
bp = ax.boxplot(data_x, patch_artist=True, vert=True, showmeans=True, 
                        meanprops={"marker": "*",
                       "markeredgecolor": "black",
                       "markersize": "8",
                       "markerfacecolor": "black"},
                        medianprops = {"color" : "black"},
                        )

# Change boxplot colors
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)


# Violin plot data
violin_colors = [orange2, orange2, orange2]
vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=True)

# Modify violin plot appearance
for idx, b in enumerate(vp['bodies']):
    m = np.mean(b.get_paths()[0].vertices[:, 1])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx + 1, idx + 2)
    b.set_color(violin_colors[idx])
    b.set_alpha(1)

# Scatter plot data
for idx, (features, choice) in enumerate(zip(data_x, all_choices)):
    # Add jitter effect so points don't overlap
    x = np.full(len(features), idx + 0.8)  # Fixed x-axis positions
    idxs = np.arange(len(x))
    out = x.astype(float)
    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
    x = out

    # Assign colors based on choice values using custom_palette
    scatter_colors = [custom_palette[val] for val in choice]
    plt.scatter(x, features, s=10, c=scatter_colors, label=f"Feature {idx + 1}", alpha=1)

# Add labels and legend
plt.ylabel(r'$\omega$ - weighting factor', fontsize=fontsize)
plt.ylim([0,10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1, 2, 3])  
ax.set_xticklabels(["Human", "Pigeon", "Rat"], fontsize=fontsize)
ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=fontsize)
plt.savefig(os.path.join(script_dir, 'save_plots/M7_1/weight.png'), dpi=300, bbox_inches='tight')

plt.show()









'''



###########################################################################
#alternative way to plot parameters of winning model per species
###########################################################################
'''
epsilons = {}
alphas = {}
lambdas = {}
lambda_seq = {}
weights = {}
gammas = {}

all_eps = np.array([])
all_alpha = np.array([])
all_lambda = np.array([])
all_lambdaseq = np.array([])
all_weight = np.array([])
all_gamma = np.array([])
all_labels = np.array([])
all_choices = np.array([])





all_labels = np.append(all_labels, human_data['label'])
all_labels = np.append(all_labels, pigeon_data['label'])
all_labels = np.append(all_labels, rat_data['label'])

all_choices = np.append(all_choices, human_data['choices'])
all_choices = np.append(all_choices, pigeon_data['choices'])
all_choices = np.append(all_choices, rat_data['choices'])

all_eps = np.append(all_eps, human_data['eps'])
all_eps = np.append(all_eps, pigeon_data['eps'])
all_eps = np.append(all_eps, rat_data['eps'])

all_alpha = np.append(all_alpha, human_data['alpha'])
all_alpha = np.append(all_alpha, pigeon_data['alpha'])
all_alpha = np.append(all_alpha, rat_data['alpha'])

all_lambda = np.append(all_lambda, human_data['lambda'])
all_lambda = np.append(all_lambda, pigeon_data['lambda'])
all_lambda = np.append(all_lambda, rat_data['lambda'])

all_lambdaseq = np.append(all_lambdaseq, human_data['lambda_s'])
all_lambdaseq = np.append(all_lambdaseq, np.zeros(pigeon_data['lambda'].shape))
all_lambdaseq = np.append(all_lambdaseq, np.zeros(rat_data['lambda'].shape))

all_weight = np.append(all_weight, human_data['w'])
all_weight = np.append(all_weight, np.zeros(pigeon_data['lambda'].shape))
all_weight = np.append(all_weight, np.zeros(rat_data['lambda'].shape))

all_gamma = np.append(all_gamma, human_data['gamr'])
all_gamma = np.append(all_gamma, pigeon_data['gamr'])
all_gamma = np.append(all_gamma, np.zeros(rat_data['lambda'].shape))


epsilons['epsilon'] = all_eps
epsilons['labels'] = all_labels
epsilons['choices'] = all_choices

alphas['alpha'] = all_alpha
alphas['labels'] = all_labels
alphas['choices'] = all_choices

lambdas['lambda'] = all_lambda
lambdas['labels'] = all_labels
lambdas['choices'] = all_choices

lambda_seq['lambda_s'] = all_lambdaseq
lambda_seq['labels'] = all_labels
lambda_seq['choices'] = all_choices

weights['w'] = all_weight
weights['labels'] = all_labels
weights['choices'] = all_choices

gammas['gamr'] = all_gamma
gammas['labels'] = all_labels
gammas['choices'] = all_choices

fontsize = 10

df_eps = pd.DataFrame(data=epsilons)
df_alpha = pd.DataFrame(data=alphas)
df_lambda = pd.DataFrame(data=lambdas)
df_lambda_seq = pd.DataFrame(data=lambda_seq)
df_weights = pd.DataFrame(data=weights)
df_gammas = pd.DataFrame(data=gammas)

purple = [(159/255, 95/255, 207/255)]
red = [(255/255, 147/255, 147/255)]
yellow = [(255/255, 255/255, 129/255)]
orange = [(255/255, 174/255, 67/255)]
orange2 = [(249/255, 197/255, 26/255)]
turquoise = [(138/255, 222/255, 184/255)]


custom_palette = {2: "#493710", 4: "#88661E", 8: "#ddae48", 5: '#698286', 500: 'black'} 
sizeL = 3

f_eps, ax_eps = plt.subplots(1, 1,figsize=(6,3))
dx = 'labels'
dy = 'epsilon'
ax_eps=pt.half_violinplot( x = dx, y = dy, data = df_eps, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None, palette=purple)
ax_eps=sns.stripplot( x = dx, y = dy, data = df_eps,
                 size = sizeL, jitter=1, zorder = 0, hue='choices', palette=custom_palette)

plt.ylabel('epsilon',fontsize = fontsize)
plt.yticks(fontsize= fontsize)
plt.xticks(fontsize = fontsize)
plt.legend('')
#pt.RainCloud( x = 'labels', y = 'epsilon', data = df_eps, width_viol = .6, ax = ax_eps)
plt.ylim([0, 1])
plt.show()

f_alpha, ax_alpha = plt.subplots(1, 1, figsize=(6,3))
dx = 'labels'
dy = 'alpha'
ax_alpha=pt.half_violinplot( x = dx, y = dy, data = df_alpha, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None,palette=red)
ax_alpha=sns.stripplot( x = dx, y = dy, data = df_alpha,
                 size = 3, jitter = 1, zorder = 0, hue='choices',palette=custom_palette)
plt.ylabel('learning rate',fontsize = fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
plt.legend('')
#pt.RainCloud( x = 'labels', y = 'learning rate', data = df_alpha, width_viol = .6, ax = ax_alpha)
plt.ylim([0, 1])
plt.show()



f_unchosen, ax_unchosen = plt.subplots(1, 1, figsize=(6,3))
dx = 'labels'
dy = 'lambda'
ax_unchosen=pt.half_violinplot( x = dx, y = dy, data = df_lambda, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None,palette=yellow)
ax_unchosen=sns.stripplot( x = dx, y = dy, data = df_lambda,
                 size = 3, jitter = 1, zorder = 0, hue='choices', palette=custom_palette)
plt.ylabel('lambda',fontsize = fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
plt.legend('')
#pt.RainCloud( x = 'labels', y = 'unchosen option values', data = df_unchosen, width_viol = .6, ax = ax_unchosen)
plt.ylim([-0.5, 0.5])
plt.show()




f_gamma, ax_gamma = plt.subplots(1, 1, figsize=(6,3))
dx = 'labels'
dy = 'gamr'
ax_gamma=pt.half_violinplot( x = dx, y = dy, data = df_gammas, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None,palette=turquoise)
ax_gamma=sns.stripplot( x = dx, y = dy, data = df_gammas,
                 size = 3, jitter = 1, zorder = 0, hue='choices', palette=custom_palette)
plt.ylabel('gamma recent',fontsize = fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
plt.legend('')
plt.ylim([0, 1])
plt.show()




f_lambda_s, ax_lambda_s = plt.subplots(1, 1, figsize=(6,3))
dx = 'labels'
dy = 'lambda_s'
ax_lambda_s=pt.half_violinplot( x = dx, y = dy, data = df_lambda_seq, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None,palette=orange)
ax_lambda_s=sns.stripplot( x = dx, y = dy, data = df_lambda_seq,
                 size = 3, jitter = 1, zorder = 0, hue='choices', palette=custom_palette)
plt.ylabel('lambda sequence',fontsize = fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
plt.legend('')
plt.ylim([-0.5, 0.5])
plt.show()


f_weight, ax_weight = plt.subplots(1, 1, figsize=(6,3))
dx = 'labels'
dy = 'w'
ax_weight=pt.half_violinplot( x = dx, y = dy, data = df_weights, bw = .2, cut = 0.,
                      scale = "area", width = .6, inner = None,palette=orange2)
ax_weight=sns.stripplot( x = dx, y = dy, data = df_weights,
                 size = 3, jitter = 1, zorder = 0, hue='choices', palette=custom_palette)
plt.ylabel('weights',fontsize = fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
plt.legend('')
plt.ylim([0, 1])
plt.show()



'''
