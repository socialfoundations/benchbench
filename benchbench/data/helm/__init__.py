import os
import numpy as np
import pandas as pd


def load_helm(subset="accuracy"):
    assert subset in [
        "accuracy",
        "bias",
        "calibration",
        "fairness",
        "efficiency",
        "robustness",
        "summarization",
        "toxicity",
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
        data[c] = np.array([float(i) for i in data[c].values])

    for c in cols:
        if (
            "ECE" in c
            or "Representation" in c
            or "Toxic fraction" in c
            or "Stereotype" in c
            or "inference time" in c
        ):
            data[c] = -data[c]

    return data, cols


def test():
    data, cols = load_helm()
    print(data.head())
    print(cols)


if __name__ == "__main__":
    test()
