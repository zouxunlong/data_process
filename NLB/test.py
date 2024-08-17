from glob import glob
import os


audios=[os.path.basename(f)[:-4] for f in glob("/mnt/home/zoux/datasets/NLB/train/audio/*", recursive=True)]

for f in glob("/mnt/home/zoux/datasets/NLB/train/docx/*", recursive=True):
    filename=os.path.basename(f)
    if not filename[:-5] in audios:
        os.remove(f)
    