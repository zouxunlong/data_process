import os
import shutil
from datasets import load_from_disk
import fire


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort()
    return directories


def do_check(ds_path):
    ds = load_from_disk(ds_path)
    N = len(ds)
    if N < 16:
        shutil.rmtree(ds_path)
    print(f"complete checking {ds_path}: {N}", flush=True)


def main(dir):
    splits = get_all_split(dir)
    splits.sort()
    for split in splits:
        do_check(split)


if __name__ == "__main__":
    fire.Fire(main)
