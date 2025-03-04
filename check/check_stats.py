import json
from datasets import load_from_disk
import fire
import os
import pandas as pd


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort(reverse=True)
    return directories


def check_data(hf_folder: str, num_worker: int = 112):

    def map_fn(example):
        return {"audio_length": len(example["context"]["audio"]["array"])/16000}

    ds_paths = get_all_split(hf_folder)
    stats = {}

    for ds_path in ds_paths:
        if os.path.exists(os.path.join(ds_path, 'ds_stats.json')):
            print(f"Skipping {ds_path}", flush=True)
            stats[ds_path] = json.load(open(os.path.join(ds_path, 'ds_stats.json')))
            continue

        print('Checking {}'.format(ds_path), flush=True)
        ds = load_from_disk(ds_path).select_columns(["context"])
        ds = ds.map(map_fn,
                    batch_size        = 1,
                    writer_batch_size = 1,
                    remove_columns    = ds.column_names,
                    num_proc          = num_worker,
                    desc              = f"{os.path.basename(ds_path)}")

        num_of_samples    = len(ds)
        total_audio_hours = sum(ds["audio_length"])/3600
        max_audio_seconds = max(ds["audio_length"])
        min_audio_seconds = min(ds["audio_length"])

        curr_res = {
            "num_of_samples"   : num_of_samples,
            "total_audio_hours": total_audio_hours,
            "max_audio_seconds": max_audio_seconds,
            "min_audio_seconds": min_audio_seconds
        }


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

        curr_res["language_audio"]       = language_audio
        curr_res["language_instruction"] = language_instruction
        curr_res["language_answer"]      = language_answer
        curr_res["split"]                = split
        curr_res["task"]                 = task
        curr_res["dataset_name"]         = dataset_name


        with open(os.path.join(ds_path, 'ds_stats.json'), 'w') as f:
            json.dump(curr_res, f, ensure_ascii=False, indent=1)

        stats[ds_path] = curr_res

    with open(os.path.join(hf_folder, 'ds_stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('complete all', flush=True)


if __name__ == "__main__":
    fire.Fire(check_data)
