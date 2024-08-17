from datasets import Audio, Dataset
from itertools import groupby
import fire
import textgrid
import os
import pandas as pd
from glob import glob
import traceback


def fetch_speaker_metadata():

    speaker_dict = {}
    df_dict = pd.read_excel("Part_3_Speaker_Metadata.xlsx", sheet_name=['Same Room', 'Separate Room'], dtype=str)
    for i, values in enumerate(df_dict['Same Room'].fillna('Unknown').loc[0:].values):
        speaker_id = values[0].strip().capitalize()
        age = values[2].strip().capitalize()
        gender = values[1].strip().capitalize()
        education_level = values[4].strip().capitalize()
        occupation = values[5].strip().capitalize()
        ethnic_group = values[3].strip().capitalize()
        first_language = values[7].strip().capitalize()
        spoken_language = values[6].strip().capitalize()
        partner_id = values[8].strip().capitalize()
        partner_relationship = values[9].strip().capitalize()
        if speaker_id in speaker_dict.keys():
            print(i, flush=True)
        speaker_dict[speaker_id]={
            "speaker_id": speaker_id,
            "age": age,
            "gender": gender,
            "ethnic_group": ethnic_group,
            "education_level": education_level,
            "occupation": occupation,
            "first_language": first_language,
            "spoken_language": spoken_language,
            "partner_id": partner_id,
            "partner_relationship": partner_relationship,
        }
    for i, values in enumerate(df_dict['Separate Room'].fillna('Unknown').loc[0:].values):
        speaker_id = values[1].strip()
        gender = values[2].strip().capitalize()
        age = values[3].strip().capitalize()
        ethnic_group = values[4].strip().capitalize()
        education_level = values[5].strip().capitalize()
        occupation = values[6].strip().capitalize()
        first_language = values[8].strip().capitalize()
        spoken_language = values[7].strip().capitalize()
        partner_id = values[9].strip().split(".")[0]
        partner_relationship = values[10].strip().capitalize()
        if speaker_id in speaker_dict.keys():
            print(i, flush=True)
        speaker_dict[speaker_id]={
            "speaker_id": speaker_id,
            "age": age,
            "gender": gender,
            "ethnic_group": ethnic_group,
            "education_level": education_level,
            "occupation": occupation,
            "first_language": first_language,
            "spoken_language": spoken_language,
            "partner_id": partner_id,
            "partner_relationship": partner_relationship,
        }
    print('Num of speakers = {}'.format(len(speaker_dict)), flush=True)
    return speaker_dict


def process_file(key, audio_file, script_file, speaker_metadata_dict):

    conversation_id = script_file.split("/")[-1].replace("-", "_").split("_")[-2]
    speaker_id = script_file.split("/")[-1].split(".")[-2].split("_")[-1]
    room_setting = script_file.split("/")[-2]
    room_dict = {
        "Scripts_Separate": "different_room",
        "Scripts_Same": "same_room",
    }
    room_setting = room_dict[room_setting]
    settings = {"room":room_setting}

    try:
        transcriptions = textgrid.TextGrid.fromFile(script_file)
        transcription=[{"start": interval.minTime,
                        "end": interval.maxTime,
                        "sentence": interval.mark,
                        } for interval in transcriptions[0]]
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        transcription=None

    try:
        speaker_metadata=speaker_metadata_dict[speaker_id]
    except:
        print(traceback.format_exc(), flush=True)
        speaker_metadata={"speaker_id":speaker_id}

    data = {
        "audio": audio_file,
        "transcription": transcription,
        "conversation_id": conversation_id,
        "speaker": speaker_metadata,
        "settings": settings,
        "partition": "PART3",
    }
    return data


def build_hf(
        root="/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus/PART3",
        output_path="PART3.hf",
        workers=5
):

    speaker_metadata_dict = fetch_speaker_metadata()

    wav_files = glob(os.path.join(root, 'Audio_Same_CloseMic', '*.wav'), recursive=True)+\
        glob(os.path.join(root, 'Audio_Separate_IVR', '*', '*.wav'), recursive=True)+\
        glob(os.path.join(root, 'Audio_Separate_StandingMic','*.wav'), recursive=True)
    scripts = glob(os.path.join(root, 'Scripts_*', '*.TextGrid'), recursive=True)
    paths = wav_files + scripts

    def get_key(path):
        return path.split("/")[-1].split(".")[0]

    paths.sort(key=get_key)

    ds_dict = {
        "audio": [],
        "transcription": [],
        "conversation_id": [],
        "speaker": [],
        "settings": [],
        "partition": [],
    }

    for key, value in groupby(paths, key=get_key):
        try:
            files = list(value)
            if len(files) == 2:
                audio_file, script_file = files
                result = process_file(key, audio_file, script_file, speaker_metadata_dict)
                ds_dict['audio'].append(result['audio'])
                ds_dict['transcription'].append(result['transcription'])
                ds_dict['conversation_id'].append(result['conversation_id'])
                ds_dict['speaker'].append(result['speaker'])
                ds_dict['settings'].append(result['settings'])
                ds_dict['partition'].append(result['partition'])
            elif len(files) == 1 and files[0].endswith(".wav"):
                audio_file=files[0]
                script_filename = (audio_file.split("/")[-2] + "_" + audio_file.split("/")[-1]).replace(".wav", ".TextGrid")
                script_file = "{}/Scripts_Separate/{}".format(root, script_filename)
                if os.path.exists(script_file):
                    result = process_file(key, audio_file, script_file, speaker_metadata_dict)
                    ds_dict['audio'].append(result['audio'])
                    ds_dict['transcription'].append(result['transcription'])
                    ds_dict['conversation_id'].append(result['conversation_id'])
                    ds_dict['speaker'].append(result['speaker'])
                    ds_dict['settings'].append(result['settings'])
                    ds_dict['partition'].append(result['partition'])
        except Exception as e:
            print(e, flush=True)
            print(files, flush=True)

    ds = Dataset.from_dict(ds_dict)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    print(ds, flush=True)
    ds.save_to_disk(output_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(build_hf)
