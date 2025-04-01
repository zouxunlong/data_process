from json import load
from datasets import load_from_disk, concatenate_datasets
import fire
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories


def split_by_length(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/*checked_ASR", num_worker: int = 224):
    from glob import glob
    ds_paths = glob(hf_folder)

    ds_paths = [path for path in ds_paths if not "_greater_than_300" in path]

    for ds_path in ds_paths:

        print(f"start {ds_path}", flush=True)
        ds = load_from_disk(ds_path)
        if not os.path.exists(ds_path.replace("_ASR", "_30_ASR")):
            ds_30 = ds.filter(lambda batch: [audio_length < 30 for audio_length in batch["audio_length"]],
                              batched=True,
                              num_proc=num_worker)
            if len(ds_30) > 0:
                ds_30.save_to_disk(ds_path.replace("_ASR", "_30_ASR"), num_proc=4)
                print(f"complete 30s {ds_path}", flush=True)

        if not os.path.exists(ds_path.replace("_ASR", "_300_ASR")):
            ds_300 = ds.filter(lambda batch: [30 <= audio_length < 300 for audio_length in batch["audio_length"]],
                               batched=True,
                               num_proc=num_worker)
            if len(ds_300) > 0:
                ds_300.save_to_disk(ds_path.replace("_ASR", "_300_ASR"))
                print(f"complete 300s {ds_path}", flush=True)

        if not os.path.exists(ds_path+"_greater_than_300"):
            ds_greater_than_300 = ds.filter(lambda batch: [300 <= audio_length for audio_length in batch["audio_length"]],
                                            batched=True,
                                            num_proc=num_worker)
            if len(ds_greater_than_300) > 0:
                ds_greater_than_300.save_to_disk(ds_path+"_greater_than_300")
                print(f"complete greater than 300s {ds_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(split_by_length)
