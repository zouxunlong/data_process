from pprint import pprint
from datasets import load_from_disk
import tempfile
import fire
import soundfile as sf
import os

def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories

def map_fn(example):
    audio_array=example["context"]["audio"]["array"]
    fname=tempfile.NamedTemporaryFile(suffix=".opus").name
    sf.write(fname, audio_array, 16000, format='OGG', subtype='OPUS')
    example["context"]["audio"]={"bytes": open(fname, "rb").read()}
    return example

def convert(split, num_proc, dir):
    ds=load_from_disk(split)
    ds=ds.map(map_fn, num_proc=num_proc, batch_size=1, writer_batch_size=1, features=ds.features)
    if not os.path.exists(split.replace(dir, dir+"_bytes")):
        ds.save_to_disk(split.replace(dir, dir+"_bytes"), num_proc=4)

def main(dir="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal", 
         reverse=True, 
         num_proc=224
         ):
    splits=get_all_split(dir)
    splits.sort(reverse=reverse)
    pprint(splits)
    for split in splits:
        if "IMDA_PART3_30_ASR_v4" not in split:
            continue
        print("=====start {}==========".format(split), flush=True)
        if os.path.exists(split.replace(dir, dir+"_bytes")):
            print("skip {}".format(split), flush=True)
            continue
        convert(split, num_proc, dir)
        print("======complete {}=======".format(split), flush=True)

if __name__ == "__main__":
    fire.Fire(main)

