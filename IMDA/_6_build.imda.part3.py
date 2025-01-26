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
    conversation_id, audio_file, script_file1, script_file2, speaker_metadata_1, speaker_metadata_2, setting, partition = args

    textgrid_1 = textgrid.TextGrid.fromFile(script_file1)
    transcription_1 = [{"start": interval.minTime, "end": interval.maxTime,"sentence": interval.mark, } for interval in textgrid_1[0]]
    textgrid_2 = textgrid.TextGrid.fromFile(script_file2)
    transcription_2 = [{"start": interval.minTime, "end": interval.maxTime,"sentence": interval.mark, } for interval in textgrid_2[0]]

    data = {
        "audio"          : audio_file,
        "transcription1" : transcription_1,
        "transcription2" : transcription_2,
        "conversation_id": conversation_id,
        "speaker1"       : speaker_metadata_1,
        "speaker2"       : speaker_metadata_2,
        "setting"        : setting,
        "partition"      : partition,
    }
    return data


def build_ds(dict_list):
    ds=Dataset.from_list(dict_list).cast_column('audio', Audio(sampling_rate=16000))
    return ds


def main(workers=20):

    root                  = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART3"
    speaker_metadata_dict = json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/speaker_metadata_part3.json"))
    duration_dict         = json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART3/duration_dict.json", "r", encoding="utf-8"))

    wav_same     = glob(os.path.join(root, 'Audio_Same_BoundaryMic', '*.wav'), recursive=True)
    wav_separate = glob(os.path.join(root, 'Audio_Separate_mixed',  '*.wav'), recursive=True)

    params = []
    for wav in sorted(wav_same):
        wav_file        = os.path.basename(wav)
        conversation_id = wav_file.split(".")[0]
        wav_time        = duration_dict["Audio_Same_BoundaryMic"][wav_file]
        script_time1    = duration_dict["Scripts_Same"][f"{conversation_id}-1.TextGrid"]
        script_time2    = duration_dict["Scripts_Same"][f"{conversation_id}-2.TextGrid"]
        if abs(wav_time-script_time1) < 0.1 and abs(wav_time-script_time2) < 0.1:
            params.append((conversation_id, wav,
                            f"{root}/Scripts_Same/{conversation_id}-1.TextGrid",
                            f"{root}/Scripts_Same/{conversation_id}-2.TextGrid",
                            speaker_metadata_dict["same_room"][f"{conversation_id}-1"],
                            speaker_metadata_dict["same_room"][f"{conversation_id}-2"],
                            "same_room",
                            "PART3"))

    for wav in sorted(wav_separate):
        wav_file        = os.path.basename(wav)
        conversation_id = wav_file.split(".")[0]
        wav_time        = duration_dict["Audio_Separate_mixed"][wav_file]
        scripts_files   = glob(os.path.join(root, 'Scripts_Separate', f'conf_{conversation_id}_{conversation_id}_*.TextGrid'), recursive=True)
        scripts_names   = [os.path.basename(script) for script in scripts_files]
        script_time1    = duration_dict["Scripts_Separate"][scripts_names[0]]
        script_time2    = duration_dict["Scripts_Separate"][scripts_names[1]]
        if abs(wav_time-script_time1) < 0.1 and abs(wav_time-script_time2) < 0.1:
            params.append((conversation_id, wav,
                           f"{root}/Scripts_Separate/{scripts_names[0]}",
                           f"{root}/Scripts_Separate/{scripts_names[1]}",
                           speaker_metadata_dict["separate_room"][scripts_names[0][15:23]],
                           speaker_metadata_dict["separate_room"][scripts_names[1][15:23]],
                           "separate_room",
                           "PART3"))

    with Pool(processes=workers) as pool:

        results   = list(tqdm(pool.imap_unordered(get_item, params), total=len(params)))
        dict_list = results
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params     = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]
        dss        = list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds        = concatenate_datasets(dss)
    save_path = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART3"
    ds.save_to_disk(save_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(main)
