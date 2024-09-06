import shutil
import time
import fire
from datasets import load_from_disk
import os
from pprint import pprint


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def fix(split, num_proc=96):

    def filter_fn(example):
        return example['instruction']['text'].strip() not in ['No Question Found', ''] and example['answer']['text'].strip() not in ['No Answer Found', 'Template not matched.', '']

    ds = load_from_disk(split)
    N = len(ds)
    ds_filtered = ds.filter(filter_fn,
                            batch_size=1,
                            writer_batch_size=1,
                            num_proc=num_proc
                            )
    if len(ds_filtered) != N:
        new_split=split.replace("datasets_multimodal", "datasets_multimodal_new")
        ds_filtered.save_to_disk(new_split, num_proc=4)
        time.sleep(1)
        shutil.rmtree(split)
        time.sleep(1)
        os.rename(new_split, split)

def main(dir):
    splits = get_all_split(dir)
    splits.sort()
    pprint(splits)
    for split in splits:
        print("start {}".format(split), flush=True)
        fix(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
