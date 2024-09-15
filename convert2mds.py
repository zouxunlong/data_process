from streaming import MDSWriter
import numpy as np
import os
from datasets import load_from_disk
from tqdm import tqdm
from glob import glob
import shutil
from multiprocessing import Pool
from streaming.base.util import merge_index


def get_text(text):
    return text if text is not None else ""

def get_array(audio):
    return audio["array"] if audio is not None else np.array([0])


def each_task(dataset_path, dataset_output_path, dataset_length, n_cpus, task):
    chunk_size = [dataset_length//n_cpus+1 if i < dataset_length%n_cpus else dataset_length//n_cpus for i in range(n_cpus)]

    cur_start_id = 0
    for group in range(n_cpus):
        start_sample_idx = cur_start_id
        end_sample_idx = cur_start_id + chunk_size[group]
        dataset_output_subpath = os.path.join(dataset_output_path, str(group))
        yield dataset_path, dataset_output_subpath, start_sample_idx, end_sample_idx, task
        cur_start_id += chunk_size[group]

def convert_to_mds(args) -> None:
    dataset_path, dataset_output_subpath, start_sample_idx, end_sample_idx, task = args

    # A dictionary of input fields to an Encoder/Decoder type
    columns = {
        "context_text": "str",
        "context_audio": "ndarray",
        "instruction_text": "str",
        "instruction_audio": "ndarray",
        "answer_text": "str",
        "answer_audio": "ndarray",
        "task": "str"
    }

    with MDSWriter(
        out=dataset_output_subpath,
        columns=columns,
        compression="zstd",
        size_limit=1024*1024*500,
    ) as out:
        dataset = load_from_disk(dataset_path)
        for sample in tqdm(dataset.select(range(start_sample_idx, end_sample_idx))):
            try:
                out.write(
                    {
                        "context_text": get_text(sample["context"]["text"]),
                        "context_audio": get_array(sample["context"]["audio"]),
                        "instruction_text": get_text(sample["instruction"]["text"]),
                        "instruction_audio": get_array(sample["instruction"]["audio"]),
                        "answer_text": get_text(sample["answer"]["text"]),
                        "answer_audio": get_array(sample["answer"]["audio"]),
                        "task":task
                    }
                )
            except:
                pass

n_cpus      = 32
dataset_dir = "/mnt/home/zoux"
output_dir  = "/mnt/home/zoux/mds_datasets_v2"

for task in ["ASR", "ASQA", "AC", "DS", "Paralingual", "SI", "SQA", "ST"]:
    dataset_path_multimodal_test  = glob(os.path.join(dataset_dir, "datasets_multimodal/test",f"{task}/**/dataset_info.json"), recursive=True)
    dataset_path_multimodal_train = glob(os.path.join(dataset_dir, "datasets_multimodal/train",f"{task}/**/dataset_info.json"), recursive=True)
    dataset_path_nlb_test         = glob(os.path.join(dataset_dir, "nlb_data/test",f"{task}/**/dataset_info.json"), recursive=True)
    dataset_path_nlb_train        = glob(os.path.join(dataset_dir, "nlb_data/train",f"{task}/**/dataset_info.json"), recursive=True)
    dataset_path_all              = dataset_path_nlb_test + dataset_path_nlb_train + dataset_path_multimodal_test + dataset_path_multimodal_train
    # dataset_path_all              =  dataset_path_multimodal_test + dataset_path_multimodal_train
    dataset_path_all.sort()
    
    for dataset_path in dataset_path_all:

        dataset_path = os.path.dirname(dataset_path)
        dataset_base_path = dataset_path.replace(dataset_dir, "")[1:]
        dataset_output_path = os.path.join(output_dir, dataset_base_path)

        if os.path.exists(dataset_output_path):
            print(f"Skipping {dataset_output_path}", flush=True)
            continue

        print('Converting {}'.format(dataset_base_path), flush=True)
        os.makedirs(dataset_output_path, exist_ok=True)
        dataset = load_from_disk(dataset_path)
        dataset_length = len(dataset)
        arg_tuples = each_task(dataset_path, dataset_output_path, dataset_length, n_cpus, task)

        with Pool(processes=n_cpus) as pool:
            for count in pool.imap(convert_to_mds, arg_tuples):
                pass 
        merge_index(dataset_output_path, keep_local=True)
