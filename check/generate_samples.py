import json
import os
from datasets import load_from_disk
import fire
import soundfile as sf

def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def generate(split):

    ds=load_from_disk(split)
    item=ds[79]
    save_path = split.replace("/all_datasets/", "/all_datasets/samples/")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf.write("{}_{}_context.wav".format(save_path, 79), item["context"]["audio"]["array"], 16000)
    if "answer" in item.keys() and item["answer"]["audio"]:
        sf.write("{}_{}_answer.wav".format(save_path, 79), item["answer"]["audio"]["array"], 16000)
    with open("{}_{}.json".format(save_path, 79), "w", encoding="utf8") as f_out:
        del item["context"]
        if "answer" in item.keys():
            item["answer"]["audio"]=None
        f_out.write(json.dumps(item, ensure_ascii=False, indent=2))


def main(dir):
    splits=get_all_split(dir)
    splits.sort()
    for split in splits:
        generate(split)


if __name__ == "__main__":
    fire.Fire(main)
