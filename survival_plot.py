import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.utils import py_random_state
from random import choices
from scipy.stats import bernoulli
from ckp_experiments import *
from ckp import *

def main():
    p_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    survival_experiment_results = []

    num_trials = 200
    cutoff_number = 1000

    for pval in p_vals:
        for kval in k_vals:
            num_survivals = survival_experiment(N=num_trials, cutoff_num=cutoff_number, num_pars=1, p=pval, k=kval, checking='all', draw_graph=False)
            if num_survivals > 0.1 * num_trials:
                survival_experiment_results += [[pval, kval, 1]]
            else:
                survival_experiment_results += [[pval, kval, -1]]

    # Code below from https://stackoverflow.com/questions/48445892/how-to-plot-2d-data-points-with-color-according-to-third-column-value
    a = np.array(survival_experiment_results)
    mapping= {-1: ("red", "x"), 1: ("blue", "o")}

    for c in np.unique(a[:,2]):
        d = a[a[:,2] == c]
        plt.scatter(d[:,0], d[:,1], c=mapping[c][0], marker=mapping[c][1])

    plt.show()

if __name__ == "__main__":
    main()
