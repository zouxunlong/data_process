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
    fname=tempfile.NamedTemporaryFile(suffix=".ogg").name
    sf.write(fname, audio_array, 16000)
    example["context"]["audio"]={"bytes": open(fname, "rb").read()}
    return example

def convert(split):
    ds=load_from_disk(split)
    ds=ds.map(map_fn, num_proc=1, batch_size=1, writer_batch_size=1, features=ds.features)
    ds.save_to_disk(split.replace("datasets_multimodal", "datasets_multimodal_new"), num_proc=4)

def main(dir="/mnt/data/all_datasets/datasets_multimodal/other_prepared/ASR/IMDA_PART6_300_ASR_v2/train"):
    splits=get_all_split(dir)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("datasets_multimodal", "datasets_multimodal_new")):
            print("complete {}".format(split), flush=True)
            continue
        convert(split)
        print("complete {}".format(split), flush=True)

if __name__ == "__main__":
    fire.Fire(main)
    
    