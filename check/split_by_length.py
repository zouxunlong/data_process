import json
from datasets import load_from_disk
import fire
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories


def split_by_length(hf_folder: str, num_worker: int = 56):

    ds_paths = get_all_split(hf_folder)

    ds_paths=[path for path in ds_paths if not "/other/" in path]
    
    for ds_path in ds_paths:
        ds_name           = os.path.basename(ds_path).split(".")[0]
        duration          = ds_name.split("_")[-2]
        task              = ds_name.split("_")[-1]

        if duration in ["30", "60", "120", "300"]:
            continue

        print(f"start {ds_path}", flush=True)
        ds    = load_from_disk(ds_path)
        if not os.path.exists(ds_path.replace("_"+task, "_30_"+task)):
            ds_30 = ds.filter(lambda batch: [audio_length<30 for audio_length in batch["audio_length"]], batched=True, num_proc=num_worker)
            if len(ds_30)>0:
                ds_30.save_to_disk(ds_path.replace("_"+task, "_30_"+task), num_proc=8)
                print(f"complete 30s {ds_path}", flush=True)

        if not os.path.exists(ds_path.replace("_"+task, "_300_"+task)):
            ds_300 = ds.filter(lambda batch: [30<=audio_length<300 for audio_length in batch["audio_length"]], batched=True, num_proc=num_worker)
            ds_300.save_to_disk(ds_path.replace("_"+task, "_300_"+task))
            print(f"complete 300s {ds_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(split_by_length)
