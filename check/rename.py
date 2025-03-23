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


def rename(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_with_length"):

    ds_paths = get_all_split(hf_folder)

    for ds_path in ds_paths:
        ds_dir  = os.path.dirname(ds_path)
        ds_name = os.path.basename(ds_path).split(".")[0]
        if ds_name.split("_")[-1] in ["v1","v2","v3","v4","v5"]:
            new_ds_name="_".join(ds_name.split("_")[:-1])
            new_ds_name=".".join([new_ds_name] + os.path.basename(ds_path).split(".")[1:])
            new_ds_path=os.path.join(ds_dir, new_ds_name)
            os.rename(ds_path, new_ds_path)


def rename_IMDA_NLB(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_with_length"):

    ds_paths = get_all_split(hf_folder)

    for ds_path in ds_paths:
        ds_dir  = os.path.dirname(ds_path)
        ds_name = os.path.basename(ds_path).split(".")[0]
        if "IMDA_PART" in ds_name or "NLB_" in ds_name:
            if "_30_" in ds_name:
                new_ds_name=ds_name.replace("_30_", "_conv_30_")
            elif "_60_" in ds_name:
                new_ds_name=ds_name.replace("_60_", "_conv_60_")
            elif "_120_" in ds_name:
                new_ds_name=ds_name.replace("_120_", "_conv_120_")
            elif "_300_" in ds_name:
                new_ds_name=ds_name.replace("_300_", "_conv_300_")
            else:
                new_ds_name=ds_name.replace("_AR", "_mono_AR").replace("_GR", "_mono_GR").replace("_MIX", "_mono_MIX").replace("_ASR", "_mono_ASR")

            new_ds_path=os.path.join(ds_dir, new_ds_name)
            os.rename(ds_path, new_ds_path)


def rename_rest(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_with_length"):

    ds_paths = get_all_split(hf_folder)
    stats = {}

    for ds_path in ds_paths:
        ds_name           = os.path.basename(ds_path).split(".")[0]
        duration, task    = ds_name.split("_")[-2:]
        stats[ds_path]    = json.load(open(os.path.join(ds_path, 'ds_stats.json')))
        max_audio_seconds = stats[ds_path]["max_audio_seconds"]

        if max_audio_seconds<=30 and "30"!=duration:
            new_ds_path = ds_path.replace("_"+task, "_30_"+task)
            os.rename(ds_path, new_ds_path)

def move2extra(dir):
    hf_folder="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2"
    hf_extra_folder="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_extra"
    os.makedirs(os.path.dirname(dir).replace(hf_folder, hf_extra_folder), exist_ok=True)
    os.rename(dir, dir.replace(hf_folder, hf_extra_folder))

def move(dir: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2"):
    
    ds_paths = get_all_split(dir)
    
    for ds_path in ds_paths:
        ds_name  = os.path.basename(ds_path)
        duration = ds_name.split("_")[-2]
        if duration in ["30", "60", "120", "300"]:
            continue

        move2extra(ds_path)


def addup_task(dir: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2"):
    ds_paths=get_all_split(dir)
    for ds_path in ds_paths:
        task = ds_path.split("/")[-2]
        if task=="ASR" and not ds_path.split("_")[-1]==task:
            print(ds_path)
            os.rename(ds_path, ds_path+"_"+task)


if __name__ == "__main__":
    fire.Fire(addup_task)

