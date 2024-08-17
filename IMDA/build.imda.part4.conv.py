from datasets import Audio, Dataset
from tqdm import tqdm
import fire
import textgrid
import os
import pandas as pd
from glob import glob
import traceback


def fetch_script_shift(file):
    script_shift_dict = {}
    with open(file) as f_in:
        for line in f_in:
            script, shift = line.split(" : ")
            script = script.replace(".wav", ".TextGrid")
            shift = float(shift)
            script_shift_dict[script] = shift
    return script_shift_dict


def fetch_speaker_metadata():

    speaker_dict = {}
    df = pd.read_excel("Part_4_Speaker_Metadata.xlsx", header=0, dtype=str)
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


def process_file(conversation_id, audio_file, script_file1, script_file2, speaker_metadata_dict, script_shift_dict):

    conversation_id = conversation_id
    lang_setting = audio_file.split("_")[-1].split(".")[0]
    lang_dict = {
        "chn": "chinese",
        "tml": "tamil",
        "mly": "malay",
    }
    lang_setting = lang_dict[lang_setting]
    room_setting = audio_file.split("/")[-2]
    room_dict = {
        "Diff_Room_Audio": "different_room",
        "Same_Room_Audio": "same_room",
    }
    room_setting = room_dict[room_setting]
    settings = {"room": room_setting, "lang": lang_setting}

    speaker_id1 = script_file1.split("/")[-1].split("_")[2]
    speaker_id2 = script_file2.split("/")[-1].split("_")[2]

    try:
        transcriptions1 = textgrid.TextGrid.fromFile(script_file1)
        if script_file1.split("/")[-1] in script_shift_dict.keys():
            shift = script_shift_dict[script_file1.split("/")[-1]]
        else:
            shift = 0
        transcription1 = [{
            "start": interval.minTime-shift,
            "end": interval.maxTime-shift,
            "sentence": interval.mark,
        } for interval in transcriptions1[0]]
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        transcription1 = None

    try:
        transcriptions2 = textgrid.TextGrid.fromFile(script_file2)
        if script_file2.split("/")[-1] in script_shift_dict.keys():
            shift = script_shift_dict[script_file2.split("/")[-1]]
        else:
            shift = 0
        transcription2 = [{
            "start": interval.minTime-shift,
            "end": interval.maxTime-shift,
            "sentence": interval.mark,
        } for interval in transcriptions2[0]]
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        transcription2 = None

    try:
        speaker_metadata1 = speaker_metadata_dict[speaker_id1]
    except:
        print(traceback.format_exc(), flush=True)
        speaker_metadata1 = {"speaker_id": speaker_id1}
    try:
        speaker_metadata2 = speaker_metadata_dict[speaker_id2]
    except:
        print(traceback.format_exc(), flush=True)
        speaker_metadata2 = {"speaker_id": speaker_id2}

    data = {
        "audio": audio_file,
        "transcription1": transcription1,
        "transcription2": transcription2,
        "conversation_id": conversation_id,
        "speaker1": speaker_metadata1,
        "speaker2": speaker_metadata2,
        "settings": settings,
        "partition": "PART4",
    }
    return data


def build_hf(
        root="/home/user/data/data_IMDA/mixed_wav/PART4",
        output_path="PART4.conv.hf",
        workers=5
):

    ds_dict = {
        "audio": [],
        "transcription1": [],
        "transcription2": [],
        "speaker1": [],
        "speaker2": [],
        "conversation_id": [],
        "settings": [],
        "partition": [],
    }

    script_shift_dict = fetch_script_shift("mix_part4.new.log")
    speaker_metadata_dict = fetch_speaker_metadata()
    wav_files = glob(os.path.join(root, '*', '*.wav'), recursive=True)
    wav_files.sort()

    for audio_file in tqdm(wav_files):
        try:
            conversation_id = audio_file.split("/")[-1][:4]
            script_files = glob(os.path.join(
                root, '*_Scripts', 'sur_{}_*.TextGrid'.format(conversation_id)), recursive=True)
            script_file1, script_file2 = script_files
            result = process_file(conversation_id, audio_file, script_file1,
                                  script_file2, speaker_metadata_dict, script_shift_dict)
            ds_dict['audio'].append(result['audio'])
            ds_dict['transcription1'].append(result['transcription1'])
            ds_dict['transcription2'].append(result['transcription2'])
            ds_dict['speaker1'].append(result['speaker1'])
            ds_dict['speaker2'].append(result['speaker2'])
            ds_dict['conversation_id'].append(result['conversation_id'])
            ds_dict['settings'].append(result['settings'])
            ds_dict['partition'].append(result['partition'])
        except Exception as e:
            print(e, flush=True)
            print(script_files, flush=True)

    ds = Dataset.from_dict(ds_dict)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    print(ds, flush=True)
    ds.save_to_disk(output_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(build_hf)
