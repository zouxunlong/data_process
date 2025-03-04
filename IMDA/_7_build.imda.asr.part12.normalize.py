import unicodedata
from datasets import load_from_disk
from tqdm import tqdm
import os
from glob import glob
import re
from multiprocessing import Pool
from fire import Fire
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def normalize_sentence(sentence):
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
    sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


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

    for i in range(len(batch["other_attributes"])):
        id            = batch["other_attributes"][i]["id"]
        transcription = id_script_map.get(id, None)
        assert transcription, "transcription not found for id: {}".format(id)
        batch["answer"][i]["text"]= "<Speaker1>: " + normalize_sentence(transcription)

    return batch


def main(workers=20):

    for part in ["PART1", "PART2"]:
        root = f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}"

        script_files = glob(os.path.join(root, '**', '*.TXT'), recursive=True)
        script_files.sort(key=lambda path: path.split("/")[-1])

        params = script_files

        with Pool(processes=workers) as pool:
            id_script_map={}
            results=list(tqdm(pool.imap_unordered(grep_script_map, params), total=len(params)))
            for result in results:
                id_script_map.update(result)

        print("id_script_map", len(id_script_map), flush=True)

        test_path=f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/test/ASR/IMDA_{part}_ASR_v4"
        if not os.path.exists(test_path.replace("_ASR_v4", "_ASR")):
            print("test", test_path, flush=True)
            ds=load_from_disk(test_path)
            ds=ds.map(map_fn, fn_kwargs={"id_script_map":id_script_map}, num_proc=1, batched=True, batch_size=1000, writer_batch_size=1, features=ds.features)
            ds.save_to_disk(test_path.replace("_ASR_v4", "_ASR"), num_proc=4)

        # train_path=f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_ASR_v4"
        # if not os.path.exists(train_path.replace("_ASR_v4", "_ASR")):
        #     print("start", train_path, flush=True)
        #     ds=load_from_disk(train_path)
        #     ds=ds.map(map_fn, fn_kwargs={"id_script_map":id_script_map}, num_proc=8, batched=True, batch_size=2000, writer_batch_size=1, features=ds.features)
        #     ds.save_to_disk(train_path.replace("_ASR_v4", "_ASR"), num_proc=4)


if __name__ == "__main__":
    Fire(main)
