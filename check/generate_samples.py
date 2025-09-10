import json
import os
from datasets import load_from_disk
import soundfile as sf

def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories

def generate(split):
    ds=load_from_disk(split)
    for i in [1,11,21,31,41,51,61,71,81,91]:
        # breakpoint()
        item=ds[i]
        save_path = split.replace("/all_datasets/", "/all_datasets/samples/")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write("{}_{}_context.wav".format(save_path, i), item["context"]["audio"]["array"], 16000)
        if "answer" in item.keys() and item["answer"]["audio"]:
            sf.write("{}_{}_answer.wav".format(save_path, i), item["answer"]["audio"]["array"], 16000)
        with open("{}_{}.json".format(save_path, i), "w", encoding="utf8") as f_out:
            del item["context"]
            if "answer" in item.keys():
                item["answer"]["audio"] = None
            f_out.write(json.dumps(item, ensure_ascii=False, indent=2))
        with open("{}_{}.txt".format(save_path, i), "w", encoding="utf8") as f_txt:
            f_txt.write(json.dumps(item["answer"]["text"], ensure_ascii=False, indent=2))

def main():
    from glob import glob
    splits=glob("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v3/datasets_multimodal/train/ASR/IMDA_PART2_mono_en_30_ASR")
    print(len(splits), flush=True)
    splits.sort()
    for split in splits:
        generate(split)

if __name__ == "__main__":
    main()
