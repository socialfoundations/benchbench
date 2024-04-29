import torch
import numpy as np
from torch.optim import SGD

from ..utils.base import rankdata
from ..utils.metric import get_rank_diff, get_rank_variance


def appr_rank_diff(score, old_rank, use_weighted_loss=False):
    """
    Approximate the rank difference between the old rank and the new rank.

    Args:
        score(np.array): Scores for all models across all tasks.
        old_rank(np.array): Original rank.
        use_weighted_loss(bool): Whether use weighted loss.

    Returns:
        torch.Tensor: The loss.
    """
    loss = torch.zeros(1)
    for i in range(len(score)):
        for j in range(len(score)):
            if old_rank[j] < old_rank[i]:
                if use_weighted_loss:
                    # this weight would encourage larger rank distance to get changed first
                    loss = loss + (old_rank[i] - old_rank[j]) * max(
                        score[j] - score[i], 0.0
                    )
                else:
                    loss = loss + max(score[j] - score[i], 0.0)
    return loss


def get_sensitivity(
    data,
    cols,
    min_value=0.01,
    lr=1.0,
    num_steps=1000,
    stop_threshold=1e-5,
    normalize_epsilon=True,
    use_weighted_loss=None,
    return_weight=False,
    verbose=False,
):
    """
    Calculate the sensitivity for a given benchmark.

    Args:
        data(pd.DataFrame): Each row represents a model, each column represents a task.
        cols(list): The column names of the tasks.
        min_value(float): Min values for epsilon.
        lr(float): Learning rate for optimization.
        num_steps(int): Number of steps for optimization.
        stop_threshold(float): Stop if the loss change is smaller than this value.
        normalize_epsilon(bool): Whether normalize epsilon by std.
        use_weighted_loss(bool): Whether use weighted approximation loss, if None, use both and return the better one.
        return_weight(bool): Whether return alpha.
        verbose(bool): Whether output logs.

    Returns:
        tuple: If return_weight is True, return ((tau, MRC), alpha); else return (tau, MRC).
    """
    if use_weighted_loss is None:
        a = get_sensitivity(
            data,
            cols,
            min_value,
            lr,
            num_steps,
            stop_threshold,
            normalize_epsilon,
            use_weighted_loss=True,
            return_weight=True,
            verbose=verbose,
        )
        b = get_sensitivity(
            data,
            cols,
            min_value,
            lr,
            num_steps,
            stop_threshold,
            normalize_epsilon,
            use_weighted_loss=False,
            return_weight=True,
            verbose=verbose,
        )
        if return_weight:
            return a if a[0] > b[0] else b
        else:
            return max(a[0], b[0])

    data = data[cols].values
    data = torch.tensor(data)
    data_std = data.std(0)
    data = data[:, [i for i, _std in enumerate(data_std) if _std > 1e-8]]
    orig_data = data.clone()
    data = data - data.mean(0)
    data = data / data.std(0)

    old_score = orig_data.mean(1).detach().numpy()
    old_rank = rankdata(-old_score, method="average")

    weight = torch.ones(data.shape[1], requires_grad=True)

    def normalize_func(w):
        w1 = torch.softmax(w, dim=0)
        w2 = w1 + min_value / (1 - min_value)
        w3 = w2 / torch.sum(w2)
        return w3

    opt = SGD([weight], lr=lr)
    last_loss = 0x3F3F3F3F
    for step in range(num_steps):
        opt.zero_grad()
        norm_weight = normalize_func(weight)
        score = (data * norm_weight).mean(1)
        loss = appr_rank_diff(score, old_rank, use_weighted_loss=use_weighted_loss)

        if loss.item() <= 1e-8:
            break

        loss.backward()
        opt.step()
        if np.fabs(loss.item() - last_loss) < stop_threshold:
            break
        last_loss = loss.item()
        if verbose:
            print("Step %d, Loss = %.2lf" % (step, loss.item()))

    norm_weight = normalize_func(weight).detach().numpy()
    if normalize_epsilon:
        norm_weight = norm_weight / orig_data.std(0).numpy()
    norm_weight = norm_weight / norm_weight.max()
    new_score = (orig_data * norm_weight).mean(1).detach().numpy()
    new_rank = rankdata(-new_score, method="average")
    rank_diff = get_rank_diff(new_rank, old_rank)
    if return_weight:
        return rank_diff, norm_weight
    else:
        return rank_diff


def get_diversity(data, cols):
    """
    Calculate the diversity for a given benchmark.

    Args:
        data(pd.DataFrame): Each row represents a model, each column represents a task.
        cols(list): The column names of the tasks.

    Returns:
        tuple: (W, max_MRC), where max_MRC refers to max MRC over every pair of tasks.

    """
    return get_rank_variance(
        [rankdata(-data[c].values, method="average") for c in cols]
    )
