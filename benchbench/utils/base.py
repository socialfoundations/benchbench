import numpy as np


def is_int(x):
    """
    Check if a string can be converted to an integer.

    Args:
        x(str): Input string.

    Returns:
        bool: True if x can be converted to an integer, False otherwise
    """
    try:
        int(x)
        return True
    except ValueError:
        return False


def is_number(s):
    """
    Check if a string can be converted to a number.

    Args:
        s(str): Input string.

    Returns:
        bool: True if s can be converted to a number, False otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_combinations(s, k):
    """
    Generate all subsets of size k from set s.

    Args:
        s(list): List of elements to get combinations from.
        k(int): Size of each combination.

    Returns:
        list: A list of combinations, where each combination is represented as a list.
    """
    if k == 0:
        return [[]]
    elif k > len(s):
        return []
    else:
        all_combinations = []
        for i in range(len(s)):
            # For each element in the set, generate the combinations that include this element
            # and then recurse to generate combinations from the remaining elements
            element = s[i]
            remaining_elements = s[i + 1 :]
            for c in get_combinations(remaining_elements, k - 1):
                all_combinations.append([element] + c)
        return all_combinations


def rankdata(a, method="average"):
    assert method == "average", "Only average method is implemented"
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="quicksort")

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, np.fabs(arr[1:] - arr[:-1]) > 1e-8]  # this is the only change
    dense = obs.cumsum()[inv]

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    # average method
    return 0.5 * (count[dense] + count[dense - 1] + 1)
