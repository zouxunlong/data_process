import os
from datasets import load_from_disk


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def filtering(example):
    if not example["transcription2"] or not example["transcription1"]:
        return False
    return True


def main(*args, workers=80):
    for part in args:
        splits=get_all_split("/mnt/home/zoux/datasets/xunlong_working_repo/IMDA.hf/{}.hf".format(part))
        for split in splits:
            if os.path.exists(split.replace("IMDA.hf", "IMDA.new.hf")):
                continue
            ds=load_from_disk(split)
            print("start {}".format(split), flush=True)
            ds_filtered=ds.filter(filtering, num_proc=workers)
            if len(ds_filtered) == len(ds):
                continue
            ds_filtered.save_to_disk(split.replace("IMDA.hf", "IMDA.new.hf"), num_proc=8)
            print("complete {}".format(split), flush=True)


if __name__=="__main__":
    from fire import Fire
    Fire(main)
