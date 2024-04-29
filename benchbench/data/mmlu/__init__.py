import os
import pandas as pd


def load_mmlu():
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
        sep="\t",
    )
    cols = data.columns[4:]
    data[cols] = data[cols] * 100.0
    return data, cols


def test():
    data, cols = load_mmlu()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
