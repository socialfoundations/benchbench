import os
import pandas as pd


def load_bbh():
    data = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"),
        sep="\t",
    )
    cols = data.columns[6:]
    return data, cols
