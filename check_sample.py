import os
from datasets import load_from_disk
import fire
from tqdm import tqdm
import traceback

def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def do_check(ds_path):
    print(f"start processing {ds_path}", flush=True)
    try:
        ds = load_from_disk(ds_path)
    except:
        print(f"error loading {ds_path}", flush=True)
        return

    problem_ids=[]
    for i in tqdm(range(len(ds)), desc = f"checking"):
        try:
            sample=ds[i]
        except:
            # print(traceback.format_exc(), flush=True)
            problem_ids.append(i)

    if problem_ids:
        ds = ds.select([i for i in range(len(ds)) if i not in problem_ids])
        ds.save_to_disk(f"{ds_path}_v1", num_proc=4)
    print(f"complete processing {ds_path}", flush=True)


def main(dir="/mnt/data/all_datasets/datasets_multimodal/test/Paralingual/VoxCeleb1_Gender_v1"):
    splits=get_all_split(dir)
    splits.sort()
    for split in splits:
        do_check(split)
        
        
if __name__=="__main__":
    
    fire.Fire(main)