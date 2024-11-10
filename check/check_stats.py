import json
from datasets import load_from_disk
import fire
import os
import pandas as pd


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories


def check_data(hf_folder: str, num_worker: int = 224):

    def map_fn(example):
        return {"audio_length": len(example["context"]["audio"]["array"])/16000}

    splits = get_all_split(hf_folder)
    stats = {}

    for split in splits:
        if os.path.exists(os.path.join(split, 'ds_stats.json')):
            print(f"Skipping {split}", flush=True)
            stats[split] = json.load(open(os.path.join(split, 'ds_stats.json')))
            continue

        print('Checking {}'.format(split), flush=True)
        ds = load_from_disk(split).select_columns(["context"])
        ds = ds.map(map_fn,
                    batch_size=1, 
                    writer_batch_size=1,
                    remove_columns=ds.column_names, 
                    num_proc=num_worker)

        num_of_samples = len(ds)
        total_audio_hours = sum(ds["audio_length"])/3600
        max_audio_seconds = max(ds["audio_length"])
        min_audio_seconds = min(ds["audio_length"])

        curr_res = {
            "num_of_samples"   : num_of_samples,
            "total_audio_hours": total_audio_hours,
            "max_audio_seconds": max_audio_seconds,
            "min_audio_seconds": min_audio_seconds
        }

        with open(os.path.join(split, 'ds_stats.json'), 'w') as f:
            json.dump(curr_res, f, ensure_ascii=False, indent=1)

        stats[split] = curr_res

    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('complete all', flush=True)

    ds_stats = json.load(open(os.path.join(hf_folder, 'ds_stats.json')))
    dfList=[]
    for key, value in ds_stats.items():
        _, mnt, data, all_datasets, datasets_hf_bytes, datasets_multimodal, other_prepared, TASK, DATASET_SPLIT= key.split("/")

        num_of_samples= value['num_of_samples']
        total_audio_hours= value['total_audio_hours']
        max_audio_seconds= value['max_audio_seconds']
        min_audio_seconds= value['min_audio_seconds']
        path= f"/{mnt}/{data}/{all_datasets}/{datasets_hf_bytes}/{datasets_multimodal}/{other_prepared}/{TASK}/{DATASET_SPLIT}"

        dfList.append([other_prepared, TASK, DATASET_SPLIT, total_audio_hours, max_audio_seconds, min_audio_seconds, num_of_samples, path])
    df_new =  pd.DataFrame(dfList)
    df_new.to_excel(f'{hf_folder}.xlsx', index=False, header=False)



if __name__ == "__main__":
    fire.Fire(check_data)
