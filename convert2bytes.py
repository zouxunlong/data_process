from pprint import pprint
from datasets import load_from_disk
import tempfile
import fire
import soundfile as sf
import os
import shutil

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

def convert(split, num_proc):
    try:
        ds=load_from_disk(split)
    except:
        print("========================error loading {}=======================================".format(split), flush=True)
        return
    ds=ds.map(map_fn, num_proc=num_proc, batch_size=1, writer_batch_size=1, features=ds.features)
    ds.save_to_disk(split.replace("datasets_multimodal", "datasets_multimodal_opus_bytes"), num_proc=4)

def main(dir, num_proc=128):
    splits=get_all_split(dir)
    splits.sort()
    pprint(splits)
    for split in splits:
        print("=====start {}==========".format(split), flush=True)
        if os.path.exists(split.replace("datasets_multimodal", "datasets_multimodal_opus_bytes")):
            print("complete {}".format(split), flush=True)
            continue
        convert(split, num_proc)
        print("======complete {}=======".format(split), flush=True)

if __name__ == "__main__":
    fire.Fire(main)
    
    