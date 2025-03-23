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

    ds_paths = get_all_split(hf_folder)
    stats = {}

    for ds_path in ds_paths:

        if os.path.exists(os.path.join(ds_path, 'ds_stats.json')):
            print(f"Reading {ds_path}", flush=True)
            stats[ds_path] = json.load(open(os.path.join(ds_path, 'ds_stats.json')))
            continue

        print('Checking {}'.format(ds_path), flush=True)
        audio_lengths = load_from_disk(ds_path)["audio_length"]

        num_of_samples    = len(audio_lengths)
        total_audio_hours = sum(audio_lengths)/3600
        max_audio_seconds = max(audio_lengths)
        min_audio_seconds = min(audio_lengths)

        bucket=ds_path.split("_")[-2]
        curr_res = {
            "num_of_samples"   : num_of_samples,
            "total_audio_hours": total_audio_hours,
            "max_audio_seconds": max_audio_seconds,
            "min_audio_seconds": min_audio_seconds,
            "length_bucket"    : int(bucket)
        }

        split, task, dataset_name = ds_path.split('/')[-3:]
        if split in ["train", "test", "other_prepared"]:

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

                elif "gigaspeech2_id" in dataset_name:
                    language_audio       = ["id"]
                    language_instruction = ["en"]
                    language_answer      = ["id"]

                elif "gigaspeech2_vi" in dataset_name:
                    language_audio       = ["vi"]
                    language_instruction = ["en"]
                    language_answer      = ["vi"]

                elif "gigaspeech2_th" in dataset_name:
                    language_audio       = ["th"]
                    language_instruction = ["en"]
                    language_answer      = ["th"]

                elif "chinese_asr_new" in dataset_name:
                    language_audio       = ["zh"]
                    language_instruction = ["en"]
                    language_answer      = ["zh"]

                elif "google_resource_crowdsourced_ta" in dataset_name:
                    language_audio       = ["ta"]
                    language_instruction = ["en"]
                    language_answer      = ["ta"]

                elif "magicdata_mandarin_chinese_read_speech_corpus_zh" in dataset_name:
                    language_audio       = ["zh"]
                    language_instruction = ["en"]
                    language_answer      = ["zh"]

                elif "malay-conversational-speech-corpus_ms" in dataset_name:
                    language_audio       = ["ms"]
                    language_instruction = ["en"]
                    language_answer      = ["ms"]

                elif "Malaya-speech-malay-stt_ms" in dataset_name:
                    language_audio       = ["ms"]
                    language_instruction = ["en"]
                    language_answer      = ["ms"]

                elif "mile_tamil_asr_corpus_new_ta" in dataset_name:
                    language_audio       = ["ta"]
                    language_instruction = ["en"]
                    language_answer      = ["ta"]

                elif "SEAME" in dataset_name:
                    language_audio       = ["en", "zh"]
                    language_instruction = ["en"]
                    language_answer      = ["en", "zh"]

                elif "wenetspeech_zh" in dataset_name:
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

                elif "CVSS" in dataset_name:
                    lang_src, lang_tgt = dataset_name.split(".")[1].split("_")[1:3]

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
