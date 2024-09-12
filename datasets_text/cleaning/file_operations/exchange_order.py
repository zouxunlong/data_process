import os
import time


start_time = time.time()


def exhange_order(filepath):
    with open(filepath, 'r', encoding='utf8') as fIN, open(filepath+'.en-ta', 'w', encoding='utf8') as fOUT:
        for line in fIN:
            sentences = line.split('|||')
            if len(sentences) != 2:
                continue
            fOUT.write("{} | {}\n".format(sentences[1].strip().replace(
                '|', ''), sentences[0].strip().replace('|', '')))


rootdir = '/home/xuanlong/dataclean/data/MCI/en-ta/batch10/TD/MCI/Completed MCI_TA2EN'
for root, dirs, files in os.walk(rootdir):
    for file in files:
        exhange_order(os.path.join(root, file))

print("--- %s seconds ---" % (time.time() - start_time))
