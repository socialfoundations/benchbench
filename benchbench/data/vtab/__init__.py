import os
import pandas as pd


def load_vtab():
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
        sep="\t",
    )
    cols = data.columns[1:]
    return data, cols


def test():
    data, cols = load_vtab()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
