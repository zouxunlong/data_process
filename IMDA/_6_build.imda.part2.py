import random
from datasets import Audio, Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
import os
from glob import glob
import json
from multiprocessing import Pool
from fire import Fire
import logging
import tempfile
import soundfile as sf


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def grep_script_map(file):
    id_script_map = {}
    with open(file) as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                id, script = line.split("\t", 1)
                id= id.strip() 
                if len(id) != 9:
                    id = id[-9:]
                transcription = script.strip()
                id_script_map[id] = transcription
    return id_script_map

def process_script_file(args):
    script_file, speaker_metadata_dict, root=args
    dict_list = []
    script_map = grep_script_map(script_file)
    for id, script in script_map.items():
        channel  = id[0]
        speaker  = id[1:5]
        session  = id[5]
        wav_file = os.path.join(
            root, 
            f'DATA/CHANNEL{channel}/WAVE/SPEAKER{speaker}/SESSION{session}/{id}.WAV'
        )

        if not os.path.exists(wav_file):
            logging.info("{} not exists.".format(wav_file))
            continue

        try:
            speaker_metadata = speaker_metadata_dict[speaker]
        except KeyError:
            logging.info(f"Speaker {speaker} not found in metadata.")
            continue

        dict_list.append({
            "audio"        : wav_file,
            "transcription": script,
            "speaker"      : speaker_metadata,
            "id"           : id,
            "channel"      : channel,
            "session"      : session,
            "partition"    : "PART2",
        })
    return dict_list

def map_fn(example):
    audio_array=example["audio"]["array"]
    fname=tempfile.NamedTemporaryFile(suffix=".opus").name
    sf.write(fname, audio_array, 16000, format='OGG', subtype='OPUS')
    example["audio"]={"bytes": open(fname, "rb").read()}
    return example

def build_ds(dict_list):
    ds=Dataset.from_list(dict_list).cast_column('audio', Audio(sampling_rate=16000))
    return ds


def main(workers=20):

    root="/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART2"
    speaker_metadata_dict = json.load(open("/data/projects/13003558/zoux/workspaces/data_process/IMDA/speaker_metadata_part2.json"))

    script_files = glob(os.path.join(root, '**', '*.TXT'), recursive=True)
    script_files.sort(key=lambda path: path.split("/")[-1])

    params=[(script_file, speaker_metadata_dict, root) for script_file in script_files]

    with Pool(processes=workers) as pool:
        results=list(tqdm(pool.imap_unordered(process_script_file, params), total=len(params)))
        dict_list = [item for sublist in results for item in sublist]
        random.shuffle(dict_list)

        batch_size = len(dict_list) // (workers*2) + 1
        params = [dict_list[i*batch_size:(i+1)*batch_size] for i in range(workers*2)]
        dss=list(tqdm(pool.imap_unordered(build_ds, params), total=len(params)))

    ds=concatenate_datasets(dss)
    save_path = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART2"
    save_path_bytes = "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART2_bytes"
    
    ds.save_to_disk(save_path, num_proc=workers)

    ds=load_from_disk(save_path)
    ds=ds.map(map_fn, num_proc=112, batch_size=1, writer_batch_size=1, features=ds.features)
    ds.save_to_disk(save_path_bytes, num_proc=workers)


if __name__ == "__main__":
    Fire(main)
