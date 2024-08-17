from datasets import Audio, Dataset
import fire
from tqdm import tqdm
import os
import pandas as pd
from glob import glob


def fetch_speaker_metadata():

    speaker_dict = {}
    df1 = pd.read_excel("Speaker_Information_(Part_1).XLSX", dtype=str)
    df2 = pd.read_excel("Speaker_Information_(Part_2).XLSX", dtype=str)
    for i, values in enumerate(df1.fillna('Unknown').loc[0:].values):
        speaker_id = values[0].strip().capitalize()
        part1_id = values[1].strip().capitalize()
        part2_id = values[2].strip().capitalize()
        gender = values[3].strip().capitalize()
        ethnic_group = values[4].strip().capitalize()
        device_c0 = values[5].strip().capitalize()
        device_c1 = values[6].strip().capitalize()
        device_c2 = values[7].strip().capitalize()

        speaker_dict[part2_id] = {
            "speaker_id": speaker_id,
            "part1_id": part1_id,
            "part2_id": part2_id,
            "gender": gender,
            "ethnic_group": ethnic_group,
            "device_c0": device_c0,
            "device_c1": device_c1,
            "device_c2": device_c2,
        }
    for i, values in enumerate(df2.fillna('Unknown').loc[0:].values):
        speaker_id = values[0].strip().capitalize()
        part1_id = values[1].strip().capitalize()
        part2_id = values[2].strip().capitalize()
        gender = values[3].strip().capitalize()
        ethnic_group = values[4].strip().capitalize()
        device_c0 = values[5].strip().capitalize()
        device_c1 = values[6].strip().capitalize()
        device_c2 = values[7].strip().capitalize()

        speaker_dict[part2_id] = {
            "speaker_id": speaker_id,
            "part1_id": part1_id,
            "part2_id": part2_id,
            "gender": gender,
            "ethnic_group": ethnic_group,
            "device_c0": device_c0,
            "device_c1": device_c1,
            "device_c2": device_c2,
        }
    print('Num of speakers = {}'.format(len(speaker_dict)), flush=True)
    return speaker_dict


def grep_script_map(file):
    id_script_map = {}
    with open(file) as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                id, script = line.split("\t", 1)
                if len(id) != 9:
                    id = id[1:]  # unicode exists
                id_script_map[id.strip()] = script.strip()
    return id_script_map


def process_file(root, script_file, speaker_metadata_dict, ds_dict):

    script_map = grep_script_map(script_file)

    for id, script in script_map.items():
        channel = id[0]
        speaker_id = id[1:5]
        session = id[5]
        wav_file = os.path.join(
            root, 'DATA/CHANNEL{}/WAVE/SPEAKER{}/SESSION{}/{}.WAV'.format(channel, speaker_id, session, id))

        if not os.path.exists(wav_file):
            print("{} not exists.".format(wav_file))
            continue

        try:
            speaker_metadata = speaker_metadata_dict[speaker_id]
        except:
            speaker_metadata = {"speaker_id": speaker_id}

        ds_dict["audio"].append(wav_file)
        ds_dict["transcription"].append(script)
        ds_dict["conversation_id"].append(id)
        ds_dict["settings"].append({
            "channel": "CHANNEL{}".format(channel),
            "session": "SESSION{}".format(session),
        })
        ds_dict["speaker"].append(speaker_metadata)
        ds_dict["partition"].append("PART2")

    return ds_dict


def build_hf(
        root="/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus/PART2",
        output_path="PART2.hf",
        workers=10
):

    speaker_metadata_dict = fetch_speaker_metadata()

    script_files = glob(os.path.join(root, '**', '*.TXT'), recursive=True)
    script_files.sort(key=lambda path: path.split("/")[-1])

    ds_dict = {
        "audio": [],
        "transcription": [],
        "conversation_id": [],
        "speaker": [],
        "settings": [],
        "partition": [],
    }

    for script_file in tqdm(script_files):
        try:
            ds_dict = process_file(root, script_file, speaker_metadata_dict, ds_dict)

        except Exception as e:
            print(e, flush=True)
            print(script_file, flush=True)

    ds = Dataset.from_dict(ds_dict)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    print(ds, flush=True)
    ds.save_to_disk(output_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(build_hf)
