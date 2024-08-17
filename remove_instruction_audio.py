import os
from datasets import load_from_disk
from fire import Fire


def map_fn(batch):
    for instruction in batch['instruction']:
        instruction['audio'] = None
    return batch


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def main(
    ds_paths=["/home/all_datasets/datasets_multimodal/other_prepared/SLUE_Phase_2_v1"]
    ):
    
    for ds_path in ds_paths:
        splits=get_all_split(ds_path)
        for split in splits:
            print("start {}".format(split), flush=True)
            ds=load_from_disk(split)
            ds=ds.map(map_fn, batched=True, writer_batch_size=1, num_proc=20)
            ds.save_to_disk(split.replace("_v1", "_v2"), num_proc = 4)
            print("done {}".format(split), flush=True)


if __name__ == '__main__':
    Fire(main)