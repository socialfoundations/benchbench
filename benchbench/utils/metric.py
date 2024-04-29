import numpy as np
from scipy.stats import kendalltau


def get_kendall_tau(new_rank, old_rank):
    """
    Calculate Kendall's Tau for two rankings.

    Args:
        new_rank (np.array): new ranking
        old_rank (np.array): old ranking

    Returns:
        tuple:
            float: Kendall's Tau
            float: p-value
    """
    tau, p_value = kendalltau(new_rank, old_rank)
    tau = (1 - tau) / 2
    return tau, p_value


def get_kendall_w(rankings):
    """
    Calculate Kendall's W for a list of rankings.

    Args:
        rankings(list): a list of rankings

    Returns:
        float: Kendall's W
    """
    # Ensure the input is a numpy array for easier manipulation
    rankings = np.array(rankings, dtype=int)

    # Number of subjects/items
    n = rankings.shape[1]

    # Number of rankings/raters
    m = rankings.shape[0]

    # Step 1: Calculate sum of ranks for each item across all lists
    rank_sums = np.sum(rankings, axis=0)

    # Step 2: Calculate the mean of the sum of ranks
    mean_rank_sum = np.mean(rank_sums)

    # Step 3: Calculate the sum of squared deviations from the mean sum of ranks
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)

    # Step 4: Calculate the maximum possible sum of squared deviations
    ss_max = m**2 * (n**3 - n) / 12

    # Step 5: Calculate Kendall's W
    w = ss / ss_max

    return 1 - w


def get_rank_diff(new_rank, old_rank=None):
    """
    Get the difference between two ranks.

    Args:
        new_rank(np.array): new ranking
        old_rank(np.array): old ranking

    Returns:
        float: Kendall's Tau
        float: MRC (max rank change)
    """
    new_rank = np.array(new_rank)
    if old_rank is None:
        old_rank = np.arange(len(new_rank))
    else:
        old_rank = np.array(old_rank)
    if np.sum(np.abs(new_rank - old_rank)) <= 1e-8:
        return 0, 0
    tau = get_kendall_tau(new_rank, old_rank)[0]
    max_rank_change = np.max(np.fabs(new_rank - old_rank)) / (len(new_rank) - 1)
    return tau, max_rank_change


def get_rank_variance(all_new_rank):
    """
    Get the variance of all ranks.

    Args:
        all_new_rank(list): a list of all rankings

    Returns:
        float: w (Kendall's W)
        float: max_MRC (the max MRC over every pair of rankings)
    """
    all_rank_diff = []
    for i, new_rank_a in enumerate(all_new_rank):
        for j, new_rank_b in enumerate(all_new_rank):
            if j <= i:
                continue
            else:
                all_rank_diff.append(get_rank_diff(new_rank_a, new_rank_b)[1])
    max_rank_diff = np.mean(all_rank_diff)
    w = get_kendall_w(all_new_rank)

    return w, max_rank_diff


def rank2order(rank):
    """
    [Legacy code] Convert a rank to an order.
    """
    ret = np.zeros(len(rank), dtype=int)
    for old_rank, new_rank in enumerate(rank):
        ret[new_rank] = old_rank
    return ret


def order2rank(order):
    """
    [Legacy code] Convert an order to a rank.
    """
    ret = np.zeros(len(order), dtype=int)
    for new_rank, old_rank in enumerate(order):
        ret[old_rank] = new_rank
    return ret


def get_order_diff(new_order, old_order=None):
    """
    [Legacy code] Get the difference between two orders.
    """
    if old_order is None:
        old_order = np.arange(len(new_order))
    return get_rank_diff(order2rank(new_order), order2rank(old_order))


def get_order_variance(all_new_order):
    """
    [Legacy code] Get the variance of all orders.
    """
    all_new_rank = [order2rank(new_order) for new_order in all_new_order]
    return get_rank_variance(all_new_rank)


def _test_kendalltau():
    # Example rankings
    rank1 = [1, 2, 3, 4, 5]
    rank2 = [5, 4, 3, 2, 1]

    # Calculate Kendall's Tau
    tau, p_value = get_kendall_tau(rank1, rank2)

    # Output the result
    print(f"Kendall's Tau: {tau}")
    print(f"p-value: {p_value}")


def _test_kendallw():
    assert (
        get_kendall_w(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ]
        )
        == 0.0
    )


if __name__ == "__main__":
    _test_kendalltau()
    _test_kendallw()
