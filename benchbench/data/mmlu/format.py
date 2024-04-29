import os
import math
import numpy as np
import pandas as pd
from datasets import load_dataset

# read top 100 model names
top_100_with_duplicate = pd.read_csv("leaderboard_raw.csv", header=None)
top_100 = []
for i in top_100_with_duplicate[0].values:
    if i not in top_100:
        top_100.append(i)
print(top_100)

# download the meta data
os.makedirs("data", exist_ok=True)
with open("data/download.sh", "w") as fout:
    fout.write("git lfs install\n")
    for i in top_100:
        cmd = "git clone git@hf.co:data/%s" % i
        fout.write(cmd + "\n")
        print(cmd)
# one must download the data manually by ``cd data; bash download.sh''
# comment the following lines if you have downloaded the data
# exit(0)

# load all model names and split names
all_model_split = []
dir_dataset = os.path.join("data")
for model_name in top_100:
    model_name = model_name[len("open-llm-leaderboard/") :]
    dir_model = os.path.join("data", model_name)
    if not os.path.isdir(dir_model):
        continue
    for split_name in os.listdir(dir_model):
        if not split_name.endswith(".parquet"):
            continue
        split_name = split_name[len("results_") : -len(".parquet")]
        all_model_split.append((model_name, split_name))
print(len(all_model_split))

# load all scores and filter broken ones
ret = []
for model_name, split_name in all_model_split:
    model = load_dataset(
        "parquet",
        data_files=os.path.join("data", model_name, "results_%s.parquet" % split_name),
        split="train",
    )["results"][0]
    tasks = [i for i in model.keys() if "hendrycksTest" in i]
    if len(tasks) != 57:
        continue
    avg = np.mean([model[c]["acc_norm"] for c in tasks])
    if math.isnan(avg):
        continue
    record = dict()
    record["model_name"] = model_name
    record["split_name"] = split_name
    record["average_score"] = avg
    record.update({c: model[c]["acc_norm"] for c in tasks})
    ret.append(record)
    print(model_name, split_name, "%.2lf" % avg)
ret = sorted(ret, key=lambda x: -x["average_score"])
ret = pd.DataFrame(ret)
ret.to_csv("calibration.tsv", sep="\t")
