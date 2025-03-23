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


def map_fn(batch):
    return {"audio_length": [len(context["audio"]["array"])/16000 for context in batch["context"]]}


def add_length(hf_folder: str, start: int, end: int, num_worker: int = 112):

    ds_paths = get_all_split(hf_folder)
    

    for ds_path in ds_paths[start:end]:
        if os.path.exists(ds_path.replace(hf_folder, hf_folder+"_with_length")):
            print(f"Skipping {ds_path}", flush=True)
            continue

        print('Adding length of {}'.format(ds_path), flush=True)
        ds = load_from_disk(ds_path)

        audio_length = ds.map(map_fn,
                              batched           = True,
                              batch_size        = 1,
                              writer_batch_size = 1,
                              num_proc          = num_worker,
                              desc              = f"{os.path.basename(ds_path)}")

        ds = ds.add_column("audio_length", audio_length["audio_length"])
        if os.path.exists(ds_path.replace(hf_folder, hf_folder+"_with_length")):
            print(f"Skipping {ds_path}", flush=True)
            continue
        ds.save_to_disk(ds_path.replace(hf_folder, hf_folder+"_with_length"), num_proc=4)


if __name__ == "__main__":
    fire.Fire(add_length)
