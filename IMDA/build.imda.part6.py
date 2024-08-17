from datasets import Audio, Dataset
from tqdm import tqdm
from itertools import groupby
import fire
import multiprocessing
import textgrid
import os
from glob import glob
import traceback


def process_file(args):
    key, audio_file, script_file = args

    conversation_id = key.split("_")[1]
    speaker_id = key.split("_")[2]
    senario_setting = key.split("_")[-1]
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
    senario_setting = senario_dict[senario_setting]

    settings = {"senario":senario_setting}

    try:
        transcriptions = textgrid.TextGrid.fromFile(script_file)
        transcription=[{"start": interval.minTime,
                        "end": interval.maxTime,
                        "sentence": interval.mark,
                        } for interval in transcriptions[0]]
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        transcription=None

    speaker_metadata={"speaker_id":speaker_id}

    data = {
        "audio": audio_file,
        "transcription": transcription,
        "conversation_id": conversation_id,
        "speaker": speaker_metadata,
        "settings": settings,
        "partition": "PART6",
    }
    return data


def build_hf(
        root="/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus_-_Additional/PART6",
        output_path="PART6.hf",
        workers=5
):
    wav_files = glob(os.path.join(root, 'Call_Centre_Design_*', 'Audio', '**', '*.wav'), recursive=True)
    scripts = glob(os.path.join(root, 'Call_Centre_Design_*', 'Scripts', '*.TextGrid'), recursive=True)
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

    pool = multiprocessing.Pool(processes=workers)
    tasks = []
    for key, value in groupby(paths, key=get_key):
        try:
            files = list(value)
            audio_file, script_file = files
            tasks.append((key, audio_file, script_file))
        except Exception as e:
            print(e, flush=True)
            print(files, flush=True)

    for result in tqdm(pool.imap(process_file, tasks), total=len(tasks), desc="Processing Files"):
        ds_dict['audio'].append(result['audio'])
        ds_dict['transcription'].append(result['transcription'])
        ds_dict['conversation_id'].append(result['conversation_id'])
        ds_dict['speaker'].append(result['speaker'])
        ds_dict['settings'].append(result['settings'])
        ds_dict['partition'].append(result['partition'])
    pool.close()
    pool.join()

    ds = Dataset.from_dict(ds_dict)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    print(ds, flush=True)
    ds.save_to_disk(output_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(build_hf)
