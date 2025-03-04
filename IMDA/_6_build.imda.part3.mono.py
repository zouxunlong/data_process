import json
import random
from datasets import Audio, Dataset, concatenate_datasets
from multiprocessing import Pool
import fire
import textgrid
import os
from glob import glob
from tqdm import tqdm
import tempfile
import soundfile as sf

def get_item(args):
    audio_file, script_file, conversation_id, speaker_metadata, setting, partition = args

    transcription_textgrid = textgrid.TextGrid.fromFile(script_file)
    transcription          = [{"start": interval.minTime, "end": interval.maxTime, "sentence": interval.mark, } for interval in transcription_textgrid[0]]

    data = {
        "audio"          : audio_file,
        "transcription"  : transcription,
        "conversation_id": conversation_id,
        "speaker"        : speaker_metadata,
        "setting"        : setting,
        "partition"      : partition,
    }
    return data


def map_fn(example):
    audio_array=example["audio"]["array"]
    fname=tempfile.NamedTemporaryFile(suffix=".opus").name
    sf.write(fname, audio_array, 16000, format='OGG', subtype='OPUS')
    example["audio"]={"bytes": open(fname, "rb").read()}
    return example


def build_ds(dict_list):
    ds = Dataset.from_list(dict_list).cast_column('audio', Audio(sampling_rate=16000))
    return ds


def main(workers=20):

    root                  = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3"
    speaker_metadata_dict = json.load(open(f"{root}/speaker_metadata_part3.json"))

    params = []

    wav_files_same = glob(os.path.join(root, 'Audio_Same_CloseMic',  '*.wav'), recursive=True)
    for wav_file in sorted(wav_files_same):
        txt_file = wav_file.replace("/Audio_Same_CloseMic/", "/txt_all/").replace(".wav", ".txt")
        mse      = float(open(txt_file).readlines()[-1].split(" || ")[-1].strip())
        if mse > 2:
            print("mse too large: ", wav_file, flush=True)
            continue

        wav_filename     = os.path.basename(wav_file)
        script_file      = wav_file.replace("/Audio_Same_CloseMic/", "/Scripts_Same/").replace(".wav", ".TextGrid")
        conversation_id  = wav_filename.split("-")[0]
        speaker_id       = wav_filename.split(".")[0]
        speaker_metadata = speaker_metadata_dict[speaker_id]

        assert os.path.exists(script_file), script_file

        params.append((wav_file,
                       script_file,
                       conversation_id,
                       speaker_metadata,
                       "same_room",
                       "PART3"))


    wav_files_separate = glob(os.path.join(root, 'Audio',  '*.wav'), recursive=True)
    for wav_file in sorted(wav_files_separate): 

        txt_file = wav_file.replace("/Audio/", "/txt_all/").replace(".wav", ".txt")
        mse      = float(open(txt_file).readlines()[-1].split(" || ")[-1].strip())
        if mse > 1:
            continue

        wav_filename     = os.path.basename(wav_file)
        script_file      = wav_file.replace("/Audio/", "/Scripts/").replace(".wav", ".TextGrid")
        conversation_id  = wav_filename.split("_")[1]
        speaker_id       = wav_filename.split(".")[0].split("_")[3]
        speaker_metadata = speaker_metadata_dict[speaker_id]

        params.append((wav_file,
                       script_file,
                       conversation_id,
                       speaker_metadata,
                       "separate_room",
                       "PART3"))


    with Pool(processes=workers) as pool:

        dict_list = list(tqdm(pool.imap_unordered(get_item, params), total=len(params)))
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]

        dss = list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds = concatenate_datasets(dss)
    save_path = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART3"
    ds.save_to_disk(save_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(main)
