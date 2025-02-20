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

def map_fn(batch, id_script_map):

    id=batch["other_attributes"][0]["id"]
    transcription=id_script_map.get(id, None)
    assert transcription, "transcription not found for id: {}".format(id)
    batch["answer"][0]["text"]=transcription

    return batch



def main(workers=20):

    for part in ["PART1", "PART2"]:
        root = f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}"

        script_files = glob(os.path.join(root, '**', '*.TXT'), recursive=True)
        script_files.sort(key=lambda path: path.split("/")[-1])

        params=script_files

        with Pool(processes=workers) as pool:
            id_script_map={}
            results=list(tqdm(pool.imap_unordered(grep_script_map, params), total=len(params)))
            for result in results:
                id_script_map.update(result)

        print("id_script_map", len(id_script_map), flush=True)


        test_path=f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/ASR/IMDA_{part}_ASR_v4"
        ds=load_from_disk(test_path)
        ds=ds.map(map_fn, fn_kwargs={"id_script_map":id_script_map}, num_proc=1, batched=True, batch_size=1, writer_batch_size=1, features=ds.features)
        ds.save_to_disk(test_path.replace("_ASR_v4", "_ASR_v5"), num_proc=4)



if __name__ == "__main__":
    Fire(main)
