import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import csv
from networkx.utils import py_random_state
from scipy.stats import bernoulli
from ckp_experiments import *
from ckp import *
from tqdm import tqdm

num_trials = 20
cutoff_number = 2000

#p_vals = np.arange(0, 1, 0.01)
p_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
k_vals = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

async def collect_experiment_data(m, checking_type, filename, ckp_type):
    fields=["p", "k", "num_survived_experiments"]
    filename_full = 'experiments/' + filename + '.csv'
    #with open(r'experiments.csv', 'a') as f:
    with open(filename_full, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(p_vals)
        for kval in k_vals:
            #k_survival_experiments = [kval]
            k_survival_experiments = []
            for pval in p_vals:
                num_survivals = await survival_experiment(N=num_trials, cutoff_num=cutoff_number, num_pars=m, p=pval, k=kval, checking=checking_type, draw_graph=False, type=ckp_type)
                k_survival_experiments += [num_survivals]
            writer.writerow(k_survival_experiments)

async def create_heatmap(filename, m, check_mechanism, ckp_type):
    data = pd.read_csv('experiments/' + filename + '.csv')
    data2=(data/20).round(2)
    #data_heatmap = data.pivot("k", "p", "num_survived_experiments")
    if m != 10:
        fig = plt.figure(figsize=(10, 10))
        r = sns.heatmap(data2, cmap="YlGnBu", yticklabels=k_vals, cbar=False)
    if m == 10:
        fig = plt.figure(figsize=(12, 10))
        r = sns.heatmap(data2, cmap="YlGnBu", yticklabels=k_vals, cbar=True)
    #r.set_title("m=" + str(m) + ", check=" + check_mechanism + ", type=" + ckp_type)
    sns.set(font_scale=3)
    r.set_title(r'$m=$' + str(m))
    r.title.set_size(30)
    output_filename = 'output_' + filename + '.jpg'
    r.set(xlabel=r'$p$', ylabel=r'$k$')
    r.xaxis.label.set_size(20)
    r.yaxis.label.set_size(20)
    r.tick_params(labelsize=20)
    plt.savefig('experiments/' + output_filename)
    #plt.show()

async def plot_experiment(m, checking_mechanism, ckp_type):
    #print("Plot for m=" + str(m) + ", checking=" + checking_mechanism)
    print("Error elimination experiments for the " + str(m) + "-parent " + str(ckp_type) + " CKP")
    file_name = 'm=' + str(m) + ', check=' + checking_mechanism + ', type=' + ckp_type #+ '.csv'
    '''f = open('experiments/' + file_name + '.csv', "w")
    f.truncate()
    f.close()
    makedata = await collect_experiment_data(m, checking_mechanism, file_name, ckp_type)'''
    heatmap = await create_heatmap(file_name, m, checking_mechanism, ckp_type)

async def main():
    m_vals = [1, 2, 3, 5, 10]
    checking_mechanisms = ['exhaustive-BFS', 'BFS-wp-p']
    types = ['simple', 'general']

    for m in tqdm(m_vals):
        for checking_mechanism in tqdm(checking_mechanisms):
            for ckp_type in tqdm(types):
                make_plot = await plot_experiment(m, checking_mechanism, ckp_type)

if __name__ == "__main__":
    asyncio.run(main())
