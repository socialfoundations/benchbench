import os
import numpy as np
import pandas as pd


def load_heim(subset="alignment_human"):
    assert subset in [
        "alignment_auto",
        "nsfw",
        "quality_auto",
        "aesthetics_auto",
        "alignment_human",
        "nudity",
        "quality_human",
        "aesthetics_human",
        "black_out",
        "originality",
    ]
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "%s.tsv" % subset),
        sep="\t",
    )
    data = data.replace("-", np.nan)
    data = data.dropna(axis=0, how="all")
    data = data.dropna(axis=1, how="all")
    cols = data.columns[2:]
    for c in cols:
        if "↓" in c:
            data[c] = -data[c]
    return data, cols


def test():
    data, cols = load_heim()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
