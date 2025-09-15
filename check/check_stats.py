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


def json2excel(json_file):
    ds_stats = json.load(open(json_file))
    dfList=[]
    for key, value in ds_stats.items():
        datasets_multimodal, split, task, dataset_name = key.split("/")[-4:]

        num_of_samples    = value['num_of_samples']
        total_audio_hours = value['total_audio_hours']
        max_audio_seconds = value['max_audio_seconds']
        min_audio_seconds = value['min_audio_seconds']
        path              = f"./{datasets_multimodal}/{split}/{task}/{dataset_name}"

        dfList.append([split, task, dataset_name, total_audio_hours, max_audio_seconds, min_audio_seconds, num_of_samples, path])
    df_new =  pd.DataFrame(dfList)
    df_new.to_excel(json_file.replace("ds_stats.json", "ds_stats.xlsx"), index=False, header=False)


def check_data(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v3"):

    ds_paths = get_all_split(hf_folder)
    stats = {}

    for ds_path in ds_paths:

        if os.path.exists(os.path.join(ds_path, 'ds_stats.json')):
            print(f"Reading {ds_path}", flush=True)

            res    = json.load(open(os.path.join(ds_path, 'ds_stats.json')))
            bucket = ds_path.split("_")[-2]

            curr_res = {
                "num_of_samples"   : res["num_of_samples"],
                "total_audio_hours": res["total_audio_hours"],
                "max_audio_seconds": res["max_audio_seconds"],
                "min_audio_seconds": res["min_audio_seconds"],
                "length_bucket"    : int(bucket)
            }
        else:
            print('Checking {}'.format(ds_path), flush=True)

            audio_lengths     = load_from_disk(ds_path)["audio_length"]
            num_of_samples    = len(audio_lengths)
            total_audio_hours = sum(audio_lengths)/3600
            max_audio_seconds = max(audio_lengths)
            min_audio_seconds = min(audio_lengths)
            bucket            = ds_path.split("_")[-2]

            curr_res = {
                "num_of_samples"   : num_of_samples,
                "total_audio_hours": total_audio_hours,
                "max_audio_seconds": max_audio_seconds,
                "min_audio_seconds": min_audio_seconds,
                "length_bucket"    : int(bucket)
            }

        split, task, dataset_name = ds_path.split('/')[-3:]
        if split in ["train", "test", "other"]:

            curr_res["split"]        = split
            curr_res["task"]         = task
            curr_res["dataset_name"] = dataset_name

            with open(os.path.join(ds_path, 'ds_stats.json'), 'w') as f:
                json.dump(curr_res, f, ensure_ascii=False, indent=1)

            stats[ds_path] = curr_res

    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    json2excel(os.path.join(hf_folder, 'ds_stats.json'))
    print('complete all', flush=True)


if __name__ == "__main__":
    fire.Fire(check_data)
