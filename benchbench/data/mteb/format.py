import os

fout = open(os.path.join(os.getcwd(), "/leaderboard.tsv"), "w")
with open(os.path.join(os.getcwd(), "/vanilla.txt"), "r") as fin:
    for i, line in enumerate(fin.readlines()):
        line = line.strip().replace("\t", " ")
        if len(line) != 0:
            fout.write(line)
        else:
            fout.write("-")
        if i % 14 == 13:
            fout.write("\n")
        else:
            fout.write("\t")
