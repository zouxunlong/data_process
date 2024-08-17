from datasets import Audio, Dataset
from tqdm import tqdm
import fire
import textgrid
import os
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


def process_file(audio_file, script_file1, script_file2, script_shift_dict):

    conversation_id, senario = audio_file.split("/")[-1].split(".")[0].split("_")
    senario_dict = {
        "hol": "holiday",
        "hot": "hotel",
        "res": "restaurant",
        "bnk": "bank",
        "tel": "telephone",
        "ins": "insurance",
        "hdb": "HDB",
        "moe": "MOE",
        "msf": "MSF",
    }
    senario_setting = senario_dict[senario]
    settings = {"senario": senario_setting}

    speaker_id1 = script_file1.split("/")[-1].split("_")[2]
    speaker_id2 = script_file2.split("/")[-1].split("_")[2]
    speaker_metadata1 = {"speaker_id": speaker_id1}
    speaker_metadata2 = {"speaker_id": speaker_id2}

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

    data = {
        "audio": audio_file,
        "transcription1": transcription1,
        "transcription2": transcription2,
        "conversation_id": conversation_id,
        "speaker1": speaker_metadata1,
        "speaker2": speaker_metadata2,
        "settings": settings,
        "partition": "PART6",
    }
    return data


def build_hf(
        root="/home/user/data/data_IMDA/mixed_wav/PART6",
        output_path="PART6.conv.hf",
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

    script_shift_dict = fetch_script_shift("mix_part6.new.log")
    wav_files = glob(os.path.join(root, 'audios', '*.wav'), recursive=True)
    wav_files.sort()

    for audio_file in tqdm(wav_files):
        try:
            conversation_id, senario = audio_file.split("/")[-1].split(".")[0].split("_")
            script_files = glob(os.path.join(root, 'scripts', 'app_{}_*_phnd_cc-{}.TextGrid'.format(conversation_id, senario)), recursive=True)
            script_file1, script_file2 = script_files
            result = process_file(audio_file, script_file1, script_file2, script_shift_dict)
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
