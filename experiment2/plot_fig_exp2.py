"""
The code uses median scores
from seven different methods.

The code for SP is available at
https://github.com/JonathanCrabbe/Symbolic-Pursuit

The code for other six methods is
available at
https://github.com/ahmedmalaa/Symbolic-Metamodeling
"""


import pickle
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="script for plotting experiment 2 results")
parser.add_argument("--base_dir", default='experiment2/', type=str, help="base dir path")
args = parser.parse_args()


with open(os.path.join(args.base_dir + 'median_smgp.pickle'), 'rb') as handle:
    temp = pickle.load(handle)

# read the median ranks for different methods for each dataset
with open(os.path.join(args.base_dir + 'median_others.pickle'), 'rb') as handle:
    benchmark_dictionary = pickle.load(handle)

benchmark_dictionary['SMGP'] = temp['SMGP']

benchmark_dictionary['DL'] = benchmark_dictionary['DeepLIFT']
del benchmark_dictionary['DeepLIFT']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))

baselines = ['SMGP', 'SM', 'SR', 'SP', 'L2X', 'SHAP', 'LIME', 'DL']

XOR_results = [benchmark_dictionary[baseline]['XOR'] for baseline in baselines]
nonlinear_additive_results = [benchmark_dictionary[baseline]['nonlinear_additive'] for baseline in baselines]
switch_results = [benchmark_dictionary[baseline]['switch'] for baseline in baselines]

axes[0].boxplot(XOR_results, medianprops=dict(linewidth=3, color='red'))
axes[0].set_title('XOR', fontsize=12)

axes[1].boxplot(nonlinear_additive_results, medianprops=dict(linewidth=3, color='red'))
axes[1].set_title('Nonlinear additive', fontsize=12)

axes[2].boxplot(switch_results, medianprops=dict(linewidth=3, color='red'))
axes[2].set_title('Switch', fontsize=12)

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(XOR_results))])
    ax.set_ylabel('Median rank', fontsize=12)

plt.setp(axes, xticks=[y + 1 for y in range(len(XOR_results))], xticklabels=baselines)

fig.savefig(os.path.join(args.base_dir + 'fig_exp2.pdf'), dpi=200, bbox_inches='tight')