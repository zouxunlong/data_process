from datasets import Audio, Dataset
from tqdm import tqdm
from itertools import groupby
import fire
import textgrid
import os
import pandas as pd
from glob import glob
import traceback



def fetch_speaker_metadata():

    speaker_dict = {}
    df = pd.read_excel("NSC_Part_4_Speaker_Metadata.xlsx", header=0, dtype=str)
    for i, values in enumerate(df.loc[0:].values):
        speaker_id = values[1].strip().capitalize()
        partner_relationship = values[2].strip().capitalize()
        age = values[3].strip().capitalize()
        gender = values[4].strip().capitalize()
        ethnic_group = values[5].strip().capitalize()
        education_level = values[6].strip().capitalize()
        occupation = values[7].strip().capitalize()
        first_language = values[8].strip().capitalize()
        dominant_language = values[9].strip().capitalize()
        spoken_language = values[10].strip().capitalize()
        partner_id = values[11].strip().capitalize()
        speaker_dict[speaker_id] = {
            "speaker_id": speaker_id,
            "partner_id": partner_id,
            "age": age,
            "gender": gender,
            "ethnic_group": ethnic_group,
            "education_level": education_level,
            "occupation": occupation,
            "first_language": first_language,
            "dominant_language": dominant_language,
            "spoken_language": spoken_language,
            "partner_relationship": partner_relationship,
        }
    print('Num of speakers = {}'.format(len(speaker_dict)), flush=True)
    return speaker_dict


def process_file(key, audio_file, script_file, speaker_metadata_dict):

    conversation_id = key.split("_")[1]
    speaker_id = key.split("_")[2]
    lang_setting = key.split("_")[-1]
    room_setting = audio_file.split("/")[-2][:-6]
    lang_dict = {
        "chn": "chinese",
        "tml": "tamil",
        "mly": "malay",
    }
    lang_setting = lang_dict[lang_setting]
    room_dict = {
        "Diff_Room": "different_room",
        "Same_Room": "same_room",
    }
    room_setting = room_dict[room_setting]
    settings = {"room":room_setting, "lang":lang_setting}

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
        "partition": "PART4",
    }
    return data


def build_hf(
        root="/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus_-_Additional/PART4",
        output_path="PART4.hf",
        workers=5
):

    speaker_metadata_dict = fetch_speaker_metadata()

    wav_files = glob(os.path.join(root, 'Codeswitching', '*_Audio', '*.wav'), recursive=True)
    scripts = glob(os.path.join(root, 'Codeswitching', '*_Scripts', '*.TextGrid'), recursive=True)
    paths = wav_files + scripts

    def get_key(path):
        return path.split("/")[-1].split(".")[0].replace("-", "_")

    paths.sort(key=get_key)

    ds_dict = {
        "audio": [],
        "transcription": [],
        "conversation_id": [],
        "speaker": [],
        "settings": [],
        "partition": [],
    }

    for key, value in tqdm(groupby(paths, key=get_key)):
        try:
            files = list(value)
            audio_file, script_file = files
            result=process_file(key, audio_file, script_file, speaker_metadata_dict)
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
