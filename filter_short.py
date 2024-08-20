import shutil
import time
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


def check_data(
    hf_folder: str = "/mnt/data/all_datasets/datasets_multimodal", 
    num_worker: int = 32
    ):

    def filter_fn(example, chunk_limit):
        return len(example["context"]["audio"]["array"])/16000 > chunk_limit/2

    splits = get_all_split(hf_folder)

    for split in splits:
        if "_30_" in split:
            chunk_limit = 30
        elif "_60_" in split:
            chunk_limit = 60
        elif "_120_" in split:
            chunk_limit = 120
        elif "_300_" in split:
            chunk_limit = 300
        else:
            continue
        
        if os.path.exists(split.replace(str(chunk_limit), str(chunk_limit+1))):
            continue

        print('Checking {}'.format(split), flush=True)
        ds = load_from_disk(split)
        ds = ds.filter(filter_fn,
                        fn_kwargs={"chunk_limit": chunk_limit},
                        batch_size=1,
                        writer_batch_size=1,
                        num_proc=num_worker)
        ds.save_to_disk(split.replace(str(chunk_limit), str(chunk_limit+1)), num_proc=16)
    print('Done filter', flush=True)
    
    time.sleep(5)


    splits = get_all_split(hf_folder)

    for split in splits:
        if "_31_" in split:
            shutil.rmtree(split.replace("_31_", "_30_"), ignore_errors=True)
            os.rename(split, split.replace("_31_", "_30_"))
        elif "_61_" in split:
            shutil.rmtree(split.replace("_61_", "_60_"), ignore_errors=True)
            os.rename(split, split.replace("_61_", "_60_"))
        elif "_121_" in split:
            shutil.rmtree(split.replace("_121_", "_120_"), ignore_errors=True)
            os.rename(split, split.replace("_121_", "_120_"))
        elif "_301_" in split:
            shutil.rmtree(split.replace("_301_", "_300_"), ignore_errors=True)
            os.rename(split, split.replace("_301_", "_300_"))
        else:
            continue

    print('Done renaming', flush=True)

if __name__ == "__main__":
    fire.Fire(check_data)
