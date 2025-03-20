from datasets import load_from_disk
import fire
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=False)
    return directories


def add_length(hf_folder: str, num_worker: int = 224):

    def map_fn(batch):
        return {"audio_length": [len(context["audio"]["array"])/16000 for context in batch["context"]]}

    ds_paths = get_all_split(hf_folder)

    for ds_path in ds_paths:
        if os.path.exists(ds_path.replace("/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2", "/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_with_length")):
            print(f"Skipping {ds_path}", flush=True)
            continue

        print('Adding length of {}'.format(ds_path), flush=True)
        ds = load_from_disk(ds_path)
        if "audio_length" in ds.column_names:
            continue
        audio_length = ds.map(map_fn,
                              batched=True,
                              batch_size=1,
                              writer_batch_size=1,
                              remove_columns=ds.column_names,
                              num_proc=num_worker,
                              desc=f"{os.path.basename(ds_path)}")
        ds = ds.add_column("audio_length", audio_length["audio_length"])
        if os.path.exists(ds_path.replace("/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2", "/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_with_length")):
            continue
        ds.save_to_disk(ds_path.replace("/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2", "/data/projects/13003558/zoux/datasets/datasets_hf_stage_MNSC_v2_with_length"), num_proc=4)


if __name__ == "__main__":
    fire.Fire(add_length)
