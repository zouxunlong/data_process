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
    directories.sort(reverse=False)
    return directories


def get_text(text):
    return text if text is not None else ""


def get_bytes(audio):
    return audio["bytes"] if audio is not None else b""


def each_task(dataset_path, dataset_output_path, dataset_length, num_pro, task, language):
    chunk_size = [dataset_length//num_pro+1 if i < dataset_length%num_pro else dataset_length//num_pro for i in range(num_pro)]

    cur_start_id = 0
    for group in range(num_pro):
        start_sample_idx = cur_start_id
        end_sample_idx = cur_start_id + chunk_size[group]
        dataset_output_subpath = os.path.join(dataset_output_path, str(group))
        yield dataset_path, dataset_output_subpath, start_sample_idx, end_sample_idx, task, language
        cur_start_id += chunk_size[group]


def convert_to_mds(args) -> None:
    dataset_path, dataset_output_subpath, start_sample_idx, end_sample_idx, task, language = args

    dataset                      = load_from_disk(dataset_path).select(range(start_sample_idx, end_sample_idx))
    features                     = dataset.features
    features['context']['audio'] = Audio(sampling_rate=16000, decode=False)
    dataset                      = dataset.cast(features=features)


    # A dictionary of input fields to an Encoder/Decoder type
    columns = {
        "context_text"     : "str",
        "context_audio"    : "bytes",
        "instruction_text" : "str",
        "instruction_audio": "bytes",
        "answer_text"      : "str",
        "answer_audio"     : "bytes",
        "task"             : "str",
        "language"         : "str"
    }

    with MDSWriter(
        out=dataset_output_subpath,
        columns=columns,
        compression="zstd",
        size_limit=1024*1024*500,
    ) as out:

        for sample in tqdm(dataset):
            try:
                out.write(
                    {
                        "context_text"     : get_text(sample["context"]["text"]),
                        "context_audio"    : get_bytes(sample["context"]["audio"]),
                        "instruction_text" : get_text(sample["instruction"]["text"]),
                        "instruction_audio": get_bytes(sample["instruction"]["audio"]),
                        "answer_text"      : get_text(sample["answer"]["text"]),
                        "answer_audio"     : get_bytes(sample["answer"]["audio"]),
                        "task"             : task,
                        "language"         : language
                    }
                )
            except:
                pass


def main(intput_dir="/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v3", 
         output_dir="/data/projects/13003558/zoux/datasets/datasets_mosaic_stage_AudioLLM_v3"):

    start_time = time.time()
    num_pro    = 64

    dataset_paths  = get_all_split(intput_dir)

    for dataset_path in dataset_paths:
        dataset_output_path = dataset_path.replace(intput_dir, output_dir)

        if os.path.exists(dataset_output_path):
            print(f"Skipping {dataset_output_path}", flush=True)
            continue

        dataset        = load_from_disk(dataset_path)
        dataset_length = len(dataset)

        if dataset_length < num_pro:
            print(f"Skipping {dataset_output_path}, too few samples", flush=True)
            continue

        print('Converting {}'.format(dataset_path), flush=True)
        os.makedirs(dataset_output_path, exist_ok=True)

        task = dataset_path.split('/')[-2]
        if task == "ASR":
            language = dataset_path.split('_')[-3]
        else:
            language = ""
        arg_tuples = each_task(dataset_path, dataset_output_path, dataset_length, num_pro, task, language)

        with Pool(processes=num_pro) as pool:
            for count in pool.imap_unordered(convert_to_mds, arg_tuples):
                pass
        merge_index(dataset_output_path, keep_local=True)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds", flush=True)


if __name__ == "__main__":
    fire.Fire(main)

