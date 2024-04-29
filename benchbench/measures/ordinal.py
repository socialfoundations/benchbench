import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
from sklearn.impute import KNNImputer

from ..utils.base import rankdata
from ..utils.metric import get_rank_diff, get_rank_variance
from ..utils.win_rate import WinningRate


def appr_rank_diff(new_win_rate, inv_indices, orig_rank):
    """
    Approximate the rank difference between the original win rate and the new win rate.

    Args:
        new_win_rate(np.array): win rate for all models
        inv_indices(list): invaraint indices
        orig_rank(np.array): original win rate for only models in inv_indices

    Returns:
        torch.Tensor: approximated loss
    """
    ret = 0.0
    for i, inv_i in enumerate(inv_indices):
        for j, inv_j in enumerate(inv_indices):
            # old_rank[i] is the original rank for inv_i
            if orig_rank[i] < orig_rank[j]:
                ret += max(new_win_rate[inv_i] - new_win_rate[inv_j], -0.01)
    return ret


def get_selected_win_rate(win_rate_matrix, w, inv_indices, do_sample=True):
    """
    Get the win rate for the selected indices.

    Args:
        win_rate_matrix(torch.Tensor): i-th row and j-th column represents the win rate of i-th model over j-th model
        w(torch.Tensor): unnormalized normalized probability for each model to be selected
        inv_indices(list): indices for L
        do_sample(bool): select models based on sampling or not

    Returns:
        tuple:
            torch.Tensor: new_win_rate
            np.array: new_indices
    """
    probs = torch.sigmoid(w)
    if do_sample:
        sampler = torch.distributions.Bernoulli(probs)
        sampled = sampler.sample() + w - w.detach()
    else:
        sampled = (probs > 0.5) + w - w.detach()
    inv = torch.tensor(
        [
            (1.0 if (j == 0.0 and i in inv_indices) else 0.0)
            for i, j in enumerate(sampled)
        ]
    )
    selected = sampled + inv
    selected_diag = torch.diag(selected)
    selected_win_rate = selected_diag @ win_rate_matrix @ selected_diag
    new_win_rate = selected_win_rate.sum(1) / selected.sum()
    new_indices = np.where(selected.detach().numpy() >= 1.0 - 1e-4)[0]

    return new_win_rate, new_indices


def get_sensitivity(
    data, cols, inv_indices=None, lr=0.01, num_step=1000, return_indices=False
):
    """
    Calculate the sensitivity for a given benchmark.

    Args:
        data(pd.DataFrame): each row represents a model, each column represents a task
        cols(list): the column names of the tasks
        inv_indices(list): indices for L, the rest will be used as L^C
        lr(float): learning rate for optimization
        num_step(int): number of steps for optimization
        return_indices(bool): whether return the indices of selected irrelevant models

    Returns:
        tuple: ((tau, MRC), indices) if return_indices is True, else (tau, MRC)
    """
    if inv_indices is None:
        inv_indices = np.arange(len(data) // 5)

    torch.manual_seed(0)
    win_rate_matrix = torch.tensor(WinningRate(data, cols).win_rate)

    orig_win_rate = win_rate_matrix[inv_indices][:, inv_indices].mean(axis=1).numpy()
    orig_rank = rankdata(-orig_win_rate, method="average")

    w = torch.zeros(len(data), requires_grad=True, dtype=torch.double)
    optimizer = Adam([w], lr=lr)
    history = []
    for episode in range(num_step):
        new_win_rate, new_indices = get_selected_win_rate(
            win_rate_matrix, w, inv_indices
        )
        loss = appr_rank_diff(new_win_rate, inv_indices, orig_rank)
        if type(loss) is float:
            break
        print("Episode %d, loss %.2lf" % (episode, loss.item()), end="\r")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        new_win_rate = (
            win_rate_matrix[new_indices][:, new_indices]
            .mean(axis=1)[inv_indices]
            .detach()
            .numpy()
        )
        new_rank = rankdata(-new_win_rate)
        rank_diff = get_rank_diff(new_rank, orig_rank)
        history.append((rank_diff, new_indices))
    print()

    new_win_rate, new_indices = get_selected_win_rate(
        win_rate_matrix, w, inv_indices, do_sample=False
    )
    new_win_rate = (
        win_rate_matrix[new_indices][:, new_indices]
        .mean(axis=1)[inv_indices]
        .detach()
        .numpy()
    )
    new_rank = rankdata(-new_win_rate, method="average")
    final_rank_diff = get_rank_diff(new_rank, orig_rank)

    if len(history) == 0:
        ret = (final_rank_diff, new_indices)
    else:
        history = sorted(history, key=lambda x: -x[0][0])
        history_best_rank_diff = history[0][0]
        history_best_indices = history[0][1]
        if final_rank_diff > history_best_rank_diff:
            ret = (final_rank_diff, new_indices)
        else:
            ret = (history_best_rank_diff, history_best_indices)
    if return_indices:
        return ret
    else:
        return ret[0]


def get_diversity(data, cols):
    """
    Calculate the diversity for a given benchmark.

    Args:
        data(pd.DataFrame): each row represents a model, each column represents a task
        cols(list): the column names of the tasks

    Returns:
        tuple: (W, max_MRC), where max_MRC refers to max MRC over every pair of tasks
    """
    imputer = KNNImputer(n_neighbors=5, weights="uniform")

    data_imputed = imputer.fit_transform(data[cols].values)
    data_imputed = pd.DataFrame(data_imputed, columns=cols)

    return get_rank_variance(
        [
            rankdata(-data_imputed[c].values, method="average")
            for c in list(cols)
            if data_imputed[c].values.dtype == "float64"
        ]
    )
