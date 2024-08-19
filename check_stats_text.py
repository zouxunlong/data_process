import json
from datasets import load_from_disk
import os
from fire import Fire

def get_all_split(root_hf):
    directories = []
    for dirpath, dirnames, filenames in os.walk(root_hf):
        if len(dirnames) == 0:
            directories.append(dirpath)
    return directories


def check_data(hf_folder : str):
    splits = get_all_split(hf_folder)
    stats = {}

    for split in splits:

        if os.path.exists(os.path.join(split, 'ds_stats.json')):
            continue
        print('Checking split {}'.format(split), flush=True)
        ds = load_from_disk(split).select_columns("tokens")
        
        tokens=ds["tokens"]

        num_of_row = len(tokens)
        total_tokens=sum(tokens)
        max_tokens=max(tokens)

        curr_res={
        "num_of_row": num_of_row,
        "max_tokens": max_tokens,
        "total_tokens": total_tokens
        }

        with open(os.path.join(split, 'ds_stats.json'), 'w') as f:
            json.dump(curr_res, f, indent=1)
        
        stats[split]=curr_res
    
    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print('complete all', flush=True)


if __name__ == "__main__":
    Fire(check_data)
