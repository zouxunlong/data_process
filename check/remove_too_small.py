import json
import shutil
import fire
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories



def remove(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2"):

    ds_paths = get_all_split(hf_folder)
    stats = {}

    for ds_path in ds_paths:
        stats = json.load(open(os.path.join(ds_path, 'ds_stats.json')))
        num_of_samples = stats["num_of_samples"]

        if num_of_samples<=50:
            shutil.rmtree(ds_path)

        
if __name__ == "__main__":
    fire.Fire(remove)

