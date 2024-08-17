
from itertools import groupby


def get_key(path):
    elements = path.split("/")[-1].replace("-", "_").split("_")
    return elements[1]+"_"+elements[-1]


lines=open("mix_part4.new.log").readlines()

lines.sort()

ref=""
for key, value in groupby(lines, key=lambda path: path[:8]):
    files = list(value)
    if len(files) == 1:
        open("mix_part4.new.new.log", "a", encoding="utf8").write(files[0])
    if len(files) == 2:
        for line in files:
            if line.split(":")[-1].strip()=="0":
                open("mix_part4.new.new.log", "a", encoding="utf8").write(line)