import time
from streaming import MDSWriter
import fire
import os
from datasets import load_from_disk, Audio
from tqdm import tqdm
from multiprocessing import Pool
from streaming.base.util import merge_index


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def get_text(text):
    return text if text is not None else ""


def get_bytes(audio):
    return audio["bytes"] if audio is not None else b""


def each_task(dataset, dataset_output_path, dataset_length, num_pro, task):
    chunk_size = [dataset_length//num_pro+1 if i < dataset_length%num_pro else dataset_length//num_pro for i in range(num_pro)]

    cur_start_id = 0
    for group in range(num_pro):
        start_sample_idx = cur_start_id
        end_sample_idx = cur_start_id + chunk_size[group]
        dataset_output_subpath = os.path.join(dataset_output_path, str(group))
        yield dataset, dataset_output_subpath, start_sample_idx, end_sample_idx, task, num_pro
        cur_start_id += chunk_size[group]


def convert_to_mds(args) -> None:
    dataset, dataset_output_subpath, start_sample_idx, end_sample_idx, task, num_pro = args


    features                     = dataset.features
    features['context']['audio'] = Audio(sampling_rate=16000, mono=True, decode=False, id=None)
    dataset                      = dataset.cast(features=features, num_proc=num_pro, keep_in_memory=True)

        
    # A dictionary of input fields to an Encoder/Decoder type
    columns = {
        "context_text": "str",
        "context_audio": "bytes",
        "instruction_text": "str",
        "instruction_audio": "bytes",
        "answer_text": "str",
        "answer_audio": "bytes",
        "task": "str"
    }

    with MDSWriter(
        out=dataset_output_subpath,
        columns=columns,
        compression="zstd",
        size_limit=1024*1024*500,
    ) as out:

        for sample in tqdm(dataset.select(range(start_sample_idx, end_sample_idx))):
            try:
                out.write(
                    {
                        "context_text": get_text(sample["context"]["text"]),
                        "context_audio": get_bytes(sample["context"]["audio"]),
                        "instruction_text": get_text(sample["instruction"]["text"]),
                        "instruction_audio": get_bytes(sample["instruction"]["audio"]),
                        "answer_text": get_text(sample["answer"]["text"]),
                        "answer_audio": get_bytes(sample["answer"]["audio"]),
                        "task":task
                    }
                )
            except:
                pass


def main(intput_dir, output_dir="mds_opus"):
    
    start_time = time.time()
    num_pro    = 16

    dataset_path_all  = get_all_split(intput_dir)
    dataset_path_all.sort(reverse=True)
    

    for dataset_path in dataset_path_all:
        dataset_output_path = os.path.join(output_dir, *dataset_path.split("/")[-4:])
        
        if os.path.exists(dataset_output_path):
            print(f"Skipping {dataset_output_path}", flush=True)
            continue

        print('Converting {}'.format(dataset_path), flush=True)
        os.makedirs(dataset_output_path, exist_ok=True)
    
        dataset                      = load_from_disk(dataset_path)
        dataset_length               = len(dataset)


        task                         = dataset_path.split('/')[-2]
        arg_tuples                   = each_task(dataset, dataset_output_path, dataset_length, num_pro, task)

        with Pool(processes=num_pro) as pool:
            for count in pool.imap(convert_to_mds, arg_tuples):
                pass 
        merge_index(dataset_output_path, keep_local=True)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds", flush=True)


if __name__ == "__main__":
    fire.Fire(main)

