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
    audio_file, script_file, conversation_id, speaker_metadata, setting, partition, shift = args

    transcription_textgrid = textgrid.TextGrid.fromFile(script_file)
    transcription          = [{"start": interval.minTime-shift, "end": interval.maxTime-shift, "sentence": interval.mark, } for interval in transcription_textgrid[0]]

    data = {
        "audio"          : audio_file,
        "transcription"  : transcription,
        "conversation_id": conversation_id,
        "speaker"        : speaker_metadata,
        "setting"        : setting,
        "partition"      : partition,
    }
    return data


def build_ds(dict_list):
    ds = Dataset.from_list(dict_list).cast_column('audio', Audio(sampling_rate=16000))
    return ds


def main(workers=20):

    root                  = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3"
    speaker_metadata_dict = json.load(open(f"{root}/speaker_metadata_part3.json"))
    duration_dict         = json.load(open(f"{root}/duration_dict.json", "r", encoding="utf-8"))
    err_shift_dict        = json.load(open(f"{root}/shift.jsonl", "r", encoding="utf-8"))

    params = []
    
    scripts_same = glob(os.path.join(root, 'Scripts_Same',  '*.TextGrid'), recursive=True)
    
    for script_file in sorted(scripts_same):
        script_filename  = os.path.basename(script_file)
        wav_filename     = script_filename.replace(".TextGrid", ".wav")
        wav_file         = os.path.join(root, 'Audio_Same_CloseMic',  wav_filename)
        conversation_id  = wav_filename.split("-")[0]
        speaker_id       = wav_filename.split(".")[0]
        wav_time         = duration_dict["Audio_Same_CloseMic"][wav_filename]
        script_time      = duration_dict["Scripts_Same"][script_filename]
        speaker_metadata = speaker_metadata_dict["same_room"][speaker_id]

        if abs(wav_time-script_time) <= 0.1:
            params.append((wav_file,
                           script_file,
                           conversation_id,
                           speaker_metadata,
                           "same_room",
                           "PART3",
                           0))
        elif err_shift_dict[wav_file]["shift1"] == err_shift_dict[wav_file]["shift2"]:
            shift = err_shift_dict[wav_file]["shift1"]
            params.append((wav_file,
                           script_file,
                           conversation_id,
                           speaker_metadata,
                           "same_room",
                           "PART3",
                           shift))

    scripts_separate = glob(os.path.join(root, 'Scripts_Separate',  '*.TextGrid'), recursive=True)

    for script_file in sorted(scripts_separate):
        script_filename  = os.path.basename(script_file)
        wav_filename     = script_filename.replace(".TextGrid", ".wav")
        wav_file         = os.path.join(root, 'Audio_Separate_StandingMic',  wav_filename)
        conversation_id  = script_filename.split("_")[1]
        speaker_id       = script_filename.split(".")[0].split("_")[3]
        wav_time         = duration_dict["Audio_Separate_StandingMic"][wav_filename]
        script_time      = duration_dict["Scripts_Separate"][script_filename]
        speaker_metadata = speaker_metadata_dict["separate_room"][speaker_id]
        
        if abs(wav_time-script_time) <= 0.1:
            params.append((wav_file,
                           script_file,
                           conversation_id,
                           speaker_metadata,
                           "separate_room",
                           "PART3",
                           0))
        elif err_shift_dict[wav_file]["shift1"] == err_shift_dict[wav_file]["shift2"]:
            shift = err_shift_dict[wav_file]["shift1"]
            params.append((wav_file,
                           script_file,
                           conversation_id,
                           speaker_metadata,
                           "separate_room",
                           "PART3",
                           shift))

    scripts_checked = glob(os.path.join(root, 'checked',  '*.TextGrid'), recursive=True)

    for script_file in sorted(scripts_checked):
        script_filename  = os.path.basename(script_file)
        wav_filename     = script_filename.replace(".TextGrid", ".wav")
        wav_file         = os.path.join(root, 'checked',  wav_filename)
        conversation_id  = script_filename.split("_")[1]
        speaker_id       = script_filename.split(".")[0].split("_")[3]
        wav_time         = duration_dict["checked"][wav_filename]
        script_time      = duration_dict["checked"][script_filename]
        speaker_metadata = speaker_metadata_dict["separate_room"][speaker_id]

        params.append((wav_file,
                       script_file,
                       conversation_id,
                       speaker_metadata,
                       "separate_room",
                       "PART3",
                       0))


    with Pool(processes=workers) as pool:

        dict_list = list(tqdm(pool.imap_unordered(get_item, params), total=len(params)))
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]
        dss = list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds = concatenate_datasets(dss)
    save_path = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_mono_hf/PART3"
    ds.save_to_disk(save_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(main)
