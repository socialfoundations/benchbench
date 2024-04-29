import os
import pandas as pd


def load_mteb():
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
        sep="\t",
    )
    orig_cols = data.columns[6:]
    ret = {}
    cols = []
    for c in orig_cols:
        col_name = c.split(" (")[0]
        num_task = int(c.split(" (")[1].split(" ")[0])
        for i in range(num_task):
            ret["{}-{}".format(col_name, i)] = data[c].values.copy()
            cols.append("{}-{}".format(col_name, i))
    data = pd.concat([data, pd.DataFrame(ret)], axis=1)

    data["average_score"] = data[cols].mean(1)
    data.sort_values(by="average_score", inplace=True, ascending=False)
    return data, cols


def test():
    data, cols = load_mteb()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
