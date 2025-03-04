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


def build_ds(dict_list):
    ds=Dataset.from_list(dict_list).cast_column('audio', Audio(sampling_rate=16000))
    return ds


def main(workers=20):

    root = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6"

    params = []

    wav_files = glob(os.path.join(root, 'Audio',  '*.wav'), recursive=True)
    for wav_file in sorted(wav_files):

        txt_file = wav_file.replace("/Audio/", "/txt_all/").replace(".wav", ".txt")
        mse      = float(open(txt_file).readlines()[-1].split(" || ")[-1].strip())
        if mse > 2:
            print("mse too large: ", wav_file, flush=True)
            continue
        
        wav_filename    = os.path.basename(wav_file)
        script_file     = wav_file.replace("/Audio/", "/Scripts/").replace(".wav", ".TextGrid")
        speaker_id      = wav_filename.split("_")[2]
        setting         = wav_filename.split(".")[0].split("-")[-1]
        conversation_id = wav_filename.split("_")[1]+"_"+wav_filename.split("_")[-1].split(".")[0]

        speaker_metadata = {"speaker_id": speaker_id}

        assert os.path.exists(script_file), script_file

        params.append((wav_file,
                       script_file,
                       conversation_id,
                       speaker_metadata,
                       setting,
                       "PART6"))


    with Pool(processes=workers) as pool:

        dict_list = list(tqdm(pool.imap_unordered(get_item, params), total=len(params)))
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]

        dss = list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds        = concatenate_datasets(dss)
    save_path = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART6"
    ds.save_to_disk(save_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(main)
