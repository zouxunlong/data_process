import json
import fire
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories


def check_data(hf_folder: str = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes"):

    splits = get_all_split(hf_folder)
    stats = {}

    for split in splits:
        if os.path.exists(os.path.join(split, 'ds_stats.json')):
            print(f"Skipping {split}", flush=True)
            # stats[split] = json.load(open(os.path.join(split, 'ds_stats.json')))
            item = json.load(open(os.path.join(split, 'ds_stats.json')))
            while "num_of_samples" not in item.keys():
                item = list(item.values())[0]
            stats[split] = item
            with open(os.path.join(split, 'ds_stats.json'), 'w') as f:
                json.dump(item, f, ensure_ascii=False, indent=1)
        else:
            print(f"====================================================no status on {split}", flush=True)


    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('complete all', flush=True)




if __name__ == "__main__":
    fire.Fire(check_data)
