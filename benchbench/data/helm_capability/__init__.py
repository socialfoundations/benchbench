import os
import numpy as np
import pandas as pd


def load_helm_capability():
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
        sep="\t",
    )
    data = data.replace("-", np.nan)
    data = data.dropna(axis=0, how="all")
    data = data.dropna(axis=1, how="all")
    cols = data.columns[2:]

    for c in cols:
        data[c] = np.array([float(i) for i in data[c].values])

    return data, cols


def test():
    data, cols = load_helm_capability()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
