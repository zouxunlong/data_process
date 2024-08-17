import json
from datasets import load_from_disk
import fire
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort()
    return directories


def check_data(hf_folder: str, num_worker: int = 32):

    def map_fn(batch):
        return {"audio_length": [len(context["audio"]["array"])/16000 for context in batch["context"]]}

    splits = get_all_split(hf_folder)
    splits.sort()
    stats = {}

    for split in splits:
        
        if os.path.exists(os.path.join(split, 'ds_stats.json')):
            stats[split] = json.load(open(os.path.join(split, 'ds_stats.json')))
            continue

        print('Checking split {}'.format(split), flush=True)
        ds = load_from_disk(split).select_columns("context")
        ds = ds.map(map_fn, 
                    batch_size=10, 
                    writer_batch_size=10,
                    batched=True, 
                    remove_columns=ds.column_names, 
                    num_proc=num_worker)

        num_of_row = len(ds)
        audio_hours = sum(ds["audio_length"])/3600
        max_audio_length = max(ds["audio_length"])

        curr_res = {
            "num_of_row": num_of_row,
            "total_audio_hours": audio_hours,
            "max_audio_length(s)": max_audio_length
        }

        with open(os.path.join(split, 'ds_stats.json'), 'w') as f:
            json.dump(curr_res, f, indent=1)

        stats[split] = curr_res

    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print('complete all', flush=True)


if __name__ == "__main__":
    fire.Fire(check_data)
