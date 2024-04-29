import os
import random

import numpy as np

import pandas as pd


def load_imagenet(*args, **kwargs):
    # Due to legacy reason, instead of refactoring the code, we just make a wrapper function like this.
    return load_data(*args, **kwargs)


def load_data(load_raw=False, seed=0, num_task=20):
    if load_raw:
        data = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "leaderboard_raw.tsv"
            ),
            sep="\t",
        )
        data = data.dropna(axis=0, how="any")
        cols = [data.columns[1]]
    else:
        data = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
            sep="\t",
        )
        data = data.sort_values(by=["acc"], ascending=False).reset_index()
        if num_task < 1000:
            assert 1000 % num_task == 0 and num_task >= 1
            cols = []
            random.seed(seed)
            np.random.seed(seed)
            size_task = 1000 // num_task
            perm = np.random.permutation(1000)
            for i in range(num_task):
                task_cols = [
                    "acc_%d" % j for j in perm[i * size_task : (i + 1) * size_task]
                ]
                data["acc_aggr_%d" % i] = data[task_cols].values.mean(1)
                cols.append("acc_aggr_%d" % i)
        else:
            cols = ["acc_%d" % i for i in range(1000)]
    return data, cols


def test():
    data, cols = load_data()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
