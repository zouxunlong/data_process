import os
from datasets import load_from_disk
from fire import Fire


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def main():

    splits = get_all_split("/mnt/home/zoux/datasets/xunlong_working_repo/IMDA.hf")
    splits.sort()

    for split in splits:
        try:
            print("start {}".format(split), flush=True)
            ds = load_from_disk(split)
            print(len(ds), flush=True)
            print(ds.column_names, flush=True)
            print("complete {}".format(split), flush=True)
        except:
            print("error on {}".format(split), flush=True)

if __name__ == '__main__':
    Fire(main)
