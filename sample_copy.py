import json
import os
from datasets import load_from_disk
from fire import Fire
import soundfile as sf


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def filter_fn(batch):
    try:
        context = batch["context"]
        return [True]
    except:
        return [False]


def main():

    splits = get_all_split(
        "/mnt/data/all_datasets/datasets_multimodal/train/SI/alpaca-gpt4-audio_v1")
    splits.sort()
    print(len(splits))

    for split in splits:
        print("start {}".format(split), flush=True)
        ds = load_from_disk(split)
        print(len(ds), flush=True)
        ds = ds.filter(filter_fn,
                       batch_size=1,
                       batched=True,
                       load_from_cache_file=False,
                       keep_in_memory=True,
                       num_proc=1
                       )
        print(ds.column_names, flush=True)
        print(len(ds), flush=True)


if __name__ == '__main__':
    Fire(main)
