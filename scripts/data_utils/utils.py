import numpy as np
import torch
from itertools import groupby, count

def discretize_cont_covs(X, cont_cov_dim):
    X_categorized = np.zeros(X.shape)
    for (col_num, col) in enumerate(np.transpose(X[:, :cont_cov_dim])):
        bins = np.linspace(torch.min(col), torch.max(col), 10)
        X_categorized[:, col_num] = np.digitize(col, bins)

    return X_categorized

def longest_seq_over_percentile(a, perc=5):
    c = count()
    counts_a, values_a = np.histogram(a)
    top_n_counts = np.sort(np.argsort(counts_a)[::-1][:(10-perc)])
    longest_seq_indxs = max((list(g) for _, g in groupby(top_n_counts, lambda x: x-next(c))), key=len)
    return values_a[longest_seq_indxs[0]], values_a[longest_seq_indxs[-1]]

def get_action_min_max_per_discret_context(X, S, A, XS_dim, 
                                            adaptive_epsilon=None, 
                                            perc_adapative_epsilon=None,
                                            interval_criterion='min_max'):
    covs = np.concatenate((X, S, A), axis=1)
    print("covs.shape: ", covs.shape)
    print("covs type: ", type(covs))
    covs_wo_action = covs[:,:XS_dim]
    unique_combos = covs_wo_action[np.unique(covs_wo_action, axis=0, return_index=True)[1]]
    print("len unique combos: ", len(unique_combos))
    row_matchs = [np.argwhere(np.all(covs_wo_action == i, axis=1)) \
                                                for i in unique_combos]
    mins = np.zeros(len(A))
    maxs = np.zeros(len(A))
    if adaptive_epsilon:
        epsilons = np.zeros(len(A))

    for rows in row_matchs:
        if interval_criterion=='min_max':
            min_value = np.min(covs[rows, -1])
            max_value = np.max(covs[rows, -1])
            mins[rows] = min_value
            maxs[rows] = max_value
        elif interval_criterion=='top_perc':
            seq = covs[rows, -1]
            mins[rows], maxs[rows] = \
                longest_seq_over_percentile(seq)
        if adaptive_epsilon:
            epsilons[rows] = \
                (maxs[rows] - mins[rows])*perc_adapative_epsilon
    # return values, reshape if needed
    if adaptive_epsilon:
        return torch.tensor(mins.reshape(-1, 1), dtype=torch.float), \
                torch.tensor(maxs.reshape(-1, 1), dtype=torch.float), \
                torch.tensor(epsilons.reshape(-1, 1), dtype=torch.float)
    else:
        return torch.tensor(mins.reshape(-1, 1), dtype=torch.float), \
                    torch.tensor(maxs.reshape(-1, 1), dtype=torch.float)
