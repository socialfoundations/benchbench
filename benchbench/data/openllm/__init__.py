import os
import pandas as pd


def load_openllm():
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
        sep="\t",
    )
    cols = data.columns[3:]
    data["average_score"] = data[cols].mean(1)
    data.sort_values(by="average_score", inplace=True, ascending=False)
    return data, cols


def test():
    data, cols = load_openllm()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
