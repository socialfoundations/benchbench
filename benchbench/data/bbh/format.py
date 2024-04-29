import os
import re

fout = open(os.path.join(os.getcwd(), "leaderboard.tsv"), "w")
with open(os.path.join(os.getcwd(), "cols.txt"), "r") as fin:
    fout.write(fin.readline() + "\n")
with open(os.path.join(os.getcwd(), "vanilla.tsv"), "r") as fin:
    new_line = ""
    for i, line in enumerate(fin.readlines()):
        if i % 5 <= 3:
            new_line += line.strip()
            new_line += "\t"
        else:
            new_line += re.sub("\s+", "\t", line)
            fout.write(new_line.rstrip() + "\n")
            new_line = ""
