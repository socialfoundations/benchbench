import os

fout = open(os.path.join(os.getcwd(), "leaderboard.tsv"), "w")
with open(os.path.join(os.getcwd(), "vanilla.txt"), "r") as fin:
    for i, line in enumerate(fin.readlines()):
        line = line.strip().replace("\t", " ")
        if len(line) != 0:
            fout.write(line.split()[0])
        else:
            continue
        if i % 10 == 9:
            fout.write("\n")
        else:
            fout.write("\t")
