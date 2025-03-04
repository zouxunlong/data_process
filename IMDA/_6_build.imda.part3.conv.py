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
    conversation_id, audio_file, script_file1, script_file2, start1, start2, speaker_metadata_1, speaker_metadata_2, setting, partition = args
    
    textgrid_1      = textgrid.TextGrid.fromFile(script_file1)
    transcription_1 = [{"start": interval.minTime + start2, "end": interval.maxTime + start2,"sentence": interval.mark} for interval in textgrid_1[0]]
    textgrid_2      = textgrid.TextGrid.fromFile(script_file2)
    transcription_2 = [{"start": interval.minTime + start1, "end": interval.maxTime + start1,"sentence": interval.mark} for interval in textgrid_2[0]]

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

    speaker_metadata_dict = json.load(open("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/speaker_metadata_part3.json"))
    items                 = json.load(open("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_shift_selected.json", "r", encoding="utf-8"))
    wav_same              = glob("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/Audio_Same/*.wav", recursive=True)
    wav_mixed             = glob("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/Audio_mixed/*.wav", recursive=True)
    params                = []

    for wav_file in sorted(wav_same):
        filename = os.path.basename(wav_file)
        key      = filename.split(".")[0]

        script_file1 = wav_file.replace("Audio_Same", "Scripts_Same").replace(".wav", "-1.TextGrid")
        script_file2 = wav_file.replace("Audio_Same", "Scripts_Same").replace(".wav", "-2.TextGrid")
        start1=0/16000
        start2=0/16000

        assert os.path.exists(script_file1), script_file1
        assert os.path.exists(script_file2), script_file2

        conversation_id = key
        setting         = "same_room"
        speaker1_id     = key + "-1"
        speaker2_id     = key + "-2"

        try:
            params.append((conversation_id,
                           wav_file,
                           script_file1,
                           script_file2,
                           start1,
                           start2,
                           speaker_metadata_dict[speaker1_id],
                           speaker_metadata_dict[speaker2_id],
                           setting,
                           "PART3"))

        except KeyError as e:
            print(e, key, flush=True)


    for wav_file in sorted(wav_mixed):
        filename = os.path.basename(wav_file)
        key      = filename.split(".")[0]

        script_file1, script_file2 = items[key].keys()
        start1=items[key][script_file1]/16000
        start2=items[key][script_file2]/16000

        assert os.path.exists(script_file1), script_file1
        assert os.path.exists(script_file2), script_file2

        conversation_id = key.split("_")[1]
        setting         = "separate_room"
        speaker1_id     = os.path.basename(script_file1).split(".")[0].split("_")[3]
        speaker2_id     = os.path.basename(script_file2).split(".")[0].split("_")[3]


        try:
            params.append((conversation_id,
                           wav_file,
                           script_file1,
                           script_file2,
                           start1,
                           start2,
                           speaker_metadata_dict[speaker1_id],
                           speaker_metadata_dict[speaker2_id],
                           setting,
                           "PART3"))
        except KeyError as e:
            print(e, key, flush=True)


    with Pool(processes=workers) as pool:

        results   = list(tqdm(pool.imap_unordered(get_item, params), total=len(params)))
        dict_list = results
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params     = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]
        dss        = list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds        = concatenate_datasets(dss)
    save_path = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_conv_hf/PART3"
    ds.save_to_disk(save_path, num_proc=workers)


if __name__ == "__main__":
    fire.Fire(main)
