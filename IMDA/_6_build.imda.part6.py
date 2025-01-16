import json
import random
from datasets import Audio, Dataset, concatenate_datasets
from multiprocessing import Pool
import fire
import textgrid
import os
from glob import glob

from tqdm import tqdm


def get_item(args):
    conversation_id, audio_file, script_file1, script_file2, setting, partition = args
    
    textgrid_1 = textgrid.TextGrid.fromFile(script_file1)
    transcription_1 = [{"start": interval.minTime, "end": interval.maxTime,"sentence": interval.mark, } for interval in textgrid_1[0]]
    textgrid_2 = textgrid.TextGrid.fromFile(script_file2)
    transcription_2 = [{"start": interval.minTime, "end": interval.maxTime,"sentence": interval.mark, } for interval in textgrid_2[0]]

    data = {
        "audio"          : audio_file,
        "transcription1" : transcription_1,
        "transcription2" : transcription_2,
        "conversation_id": conversation_id,
        "setting"        : setting,
        "partition"      : partition,
    }
    return data


def build_ds(dict_list):
    ds=Dataset.from_list(dict_list).cast_column('audio', Audio(sampling_rate=16000))
    return ds


def main(workers=20):

    root                  = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6"
    duration_dict         = json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/duration_dict.json", "r", encoding="utf-8"))

    wavs = glob(os.path.join(root, 'Audio_mixed', '*.wav'), recursive=True)

    params = []

    for wav in sorted(wavs):
        wav_file = os.path.basename(wav)
        conversation_id, topic = wav_file.split(".")[0].split("_")
        wav_time = duration_dict["Audio_mixed"][wav_file]
        scripts_files = glob(os.path.join(root, 'Scripts', f'app_{conversation_id}_*-{topic}.TextGrid'), recursive=True)
        scripts_names = [os.path.basename(script) for script in scripts_files]
        script_time1 = duration_dict["Scripts"][scripts_names[0]]
        script_time2 = duration_dict["Scripts"][scripts_names[1]]
        if abs(wav_time-script_time1) < 0.1 and abs(wav_time-script_time2) < 0.1:
            params.append((conversation_id, wav,
                        f"{root}/Scripts/{scripts_names[0]}",
                        f"{root}/Scripts/{scripts_names[1]}",
                        scripts_names[1][22:25],
                        "PART6"))
        elif script_time1==script_time2:
            params.append((conversation_id, wav,
                        f"{root}/Scripts/{scripts_names[0]}",
                        f"{root}/Scripts/{scripts_names[1]}",
                        scripts_names[1][22:25],
                        "PART6"))
        else:
            print(f"Time not match: {conversation_id}_{topic}: {script_time1}/{script_time2}/{wav_time}", flush=True)
            
    print(len(params), flush=True)

    with Pool(processes=workers) as pool:

        dict_list = list(tqdm(pool.imap_unordered(get_item, params), total=len(params)))
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]
        dss = list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds = concatenate_datasets(dss)
    save_path = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART6"
    ds.save_to_disk(save_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(main)
