import os

fout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"), "w")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vanilla.txt"), "r") as fin:
    cols = []
    helm_lite = dict()
    for i, line in enumerate(fin.readlines()):
        line = line.strip()
        if len(line) == 0:
            continue
        fout.write(line)
        if i % 12 == 11:
            fout.write("\n")
        else:
            fout.write("\t")
