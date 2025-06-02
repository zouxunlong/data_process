from enum import Flag
import os
import shutil
from datasets import load_from_disk
import fire


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def filter_fn(batch):
    try:
        context = batch["context"][0]
    except:
        return [False]
    return [True]


def do_check(ds_path, num_proc=32):
    print(f"start checking {ds_path}", flush=True)
    ds = load_from_disk(ds_path)
    N = len(ds)
    ds_filtered = ds.filter(filter_fn, batched=True, batch_size=1, writer_batch_size=1, num_proc=num_proc)
    if len(ds_filtered) != N:
        ds_filtered.save_to_disk(f"{ds_path}_filtered", num_proc=4)
        shutil.rmtree(ds_path)
        os.rename(f"{ds_path}_filtered", ds_path)
        print(f"=========================================== {ds_path} {N-len(ds_filtered)} error found and filtered", flush=True)
    else:
        print(f"complete checking {ds_path}", flush=True)


def main(dir):
    splits = get_all_split(dir)
    splits.sort()
    for split in splits:
        do_check(split)


if __name__ == "__main__":
    fire.Fire(main)
