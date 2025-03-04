import json
import fire
import os
from datasets import load_from_disk


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories


def create_samples(dir):
    dirs=get_all_split(dir)
    with open("/data/projects/13003558/zoux/samples.txt", "w") as f:
        for path in dirs:
            ds = load_from_disk(path).select_columns(["instruction", "answer"])
            f.write(path+"\n"+json.dumps(ds[0], ensure_ascii=False, indent=4)+"\n")


def check_data(hf_folder: str="/data/projects/13003558/zoux/datasets/datasets_hf_bytes"):

    ds_paths = get_all_split(hf_folder)
    stats = {}

    for ds_path in ds_paths:
        if os.path.exists(os.path.join(ds_path, 'ds_stats.json')):

            item = json.load(open(os.path.join(ds_path, 'ds_stats.json')))

            split, task, dataset_name = ds_path.split('/')[-3:]
            if split in ["train", "test"]:

                if task == "AC":
                    language_audio       = []
                    language_instruction = ["en"]
                    language_answer      = ["en"]

                if task == "AQA":
                    language_audio       = []
                    language_instruction = ["en"]
                    language_answer      = ["en"]

                if task == "ASR":
                    if "IMDA_PART4" in dataset_name:
                        language_audio       = ["en","zh","ms","ta"]
                        language_instruction = ["en"]
                        language_answer      = ["en","zh","ms","ta"]
                    elif "AIShell" in dataset_name:
                        language_audio       = ["zh"]
                        language_instruction = ["en"]
                        language_answer      = ["zh"]
                    else:
                        language_audio       = ["en"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]

                if task == "PQA":
                    if "IMDA_PART4" in dataset_name:
                        language_audio       = ["en", "zh", "ms", "ta"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]
                    else:
                        language_audio       = ["en"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]

                if task == "SDS":
                    if "IMDA_PART4" in dataset_name:
                        language_audio       = ["en", "zh", "ms", "ta"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]
                    else:
                        language_audio       = ["en"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]

                if task == "SI":

                    language_audio       = ["en"]
                    language_instruction = ["en"]
                    language_answer      = ["en"]

                if task == "SQA":

                    if "ODSQA_zh" in dataset_name:

                        language_audio       = ["zh"]
                        language_instruction = ["zh"]
                        language_answer      = ["zh"]

                    elif "IMDA_PART4" in dataset_name:

                        language_audio       = ["en", "zh", "ms", "ta"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]

                    else:

                        language_audio       = ["en"]
                        language_instruction = ["en"]
                        language_answer      = ["en"]

                if task == "ST":

                    if "peoples_speech" in dataset_name:
                        lang_src, lang_tgt = dataset_name.split("_")[2:4]

                        language_audio       = [lang_src]
                        language_instruction = ["en"]
                        language_answer      = [lang_tgt]

                    elif "gigaspeech" in dataset_name:
                        lang_src, lang_tgt = dataset_name.split("_")[1:3]

                        language_audio       = [lang_src]
                        language_instruction = ["en"]
                        language_answer      = [lang_tgt]

                    elif "common_voice_17" in dataset_name:
                        lang_src, lang_tgt = dataset_name.split("_")[3:5]

                        language_audio       = [lang_src]
                        language_instruction = ["en"]
                        language_answer      = [lang_tgt]
                    
                    else:
                        assert "CoVoST2" in dataset_name
                        lang_src, lang_tgt = dataset_name.split("_")[1:3]

                        language_audio       = [lang_src]
                        language_instruction = ["en"]
                        language_answer      = [lang_tgt]

            item["language_audio"]       = language_audio
            item["language_instruction"] = language_instruction
            item["language_answer"]      = language_answer
            item["split"]                = split
            item["task"]                 = task
            item["dataset_name"]         = dataset_name

            stats[ds_path.replace("/datasets_hf_bytes/", "/datasets_mosaic/")] = item
            with open(os.path.join(ds_path, 'ds_stats.json'), 'w') as f:
                json.dump(item, f, ensure_ascii=False, indent=1)

        else:
            print(f"====================================================no status on {split}", flush=True)


    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('complete all', flush=True)




if __name__ == "__main__":
    fire.Fire(check_data)
