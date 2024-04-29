import os
import re
import pandas as pd

fout = open(os.path.join(os.getcwd(), "leaderboard_raw.tsv"), "w")
with open(os.path.join(os.getcwd(), "vanilla.txt"), "r") as fin:
    new_line = ""
    for i, line in enumerate(fin.readlines()):
        if i % 12 <= 10:
            new_line += line.strip()
            if len(line.strip()) != 0:
                new_line += "\t"
        else:
            new_line += re.sub("\s+", "\t", line)
            fout.write(new_line.rstrip() + "\n")
            new_line = ""
fout.close()

data = pd.read_csv(os.path.join(os.getcwd(), "leaderboard_raw.tsv"), sep="\t")
data.sort_values(by=["Acc@1"], inplace=True, ascending=False)
data["Model"] = data["Weight"].apply(
    lambda t: "_".join(t.split(".")[0].split("_")[:-1]).lower()
)
# data.to_csv(os.path.join(os.getcwd(), "leaderboard_raw.tsv"), sep="\t", index=False)

with open(os.path.join(os.getcwd(), "run.sh"), "w") as fout:
    for i in range(len(data)):
        fout.write(
            f"python run_imagenet.py --model_name {data['Model'][i]} --weight_name {data['Weight'][i]}\n"
        )
