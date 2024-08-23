
from multiprocessing import Pool
from datasets import load_from_disk, Audio, Features, Value, concatenate_datasets
import random
import os
from fire import Fire
from tqdm import tqdm


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def map_fn(example):
    return {
        "context": {
            "text": example["raw_document_text"],
            "audio": example["document_audio"]
        },
        "instruction": {
            "text": example["raw_question_text"],
            "audio": None
        },
        "answer": {
            "text": example["answer_spans"]["answer"][0],
            "audio": None
        },
        "other_attributes": {
            "id"                      : example["question_id"],
            "normalized_document_text": example["normalized_document_text"],
            "normalized_question_text": example["normalized_question_text"],
            "question_speaker_id"     : example["question_speaker_id"],
            "document_speaker_id"     : example["document_speaker_id"],
            "document_id"             : example["document_id"],
        }
    }


def map2schema(split, workers=32):

    ds = load_from_disk(split)

    features = Features({
        'context'         : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction'     : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer'          : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "id"                      : ds.features["question_id"],
            "normalized_document_text": ds.features["normalized_document_text"],
            "normalized_question_text": ds.features["normalized_question_text"],
            "question_speaker_id"     : ds.features["question_speaker_id"],
            "document_speaker_id"     : ds.features["document_speaker_id"],
            "document_id"             : ds.features["document_id"],
        }
    })

    num_samples = len(ds)
    print("num_samples: ", num_samples, flush=True)

    for batch, i in enumerate(range(0, num_samples, 64000)):

        ds_slice = ds.select(range(i, min(i+64000, num_samples)))
        ds_slice = ds_slice.map(map_fn,
                                features=features,
                                remove_columns=ds.column_names,
                                num_proc=workers,
                                batch_size=1,
                                writer_batch_size=1,
                                desc=f"mapping {i}-{min(i+64000, num_samples)}"
                                )
        problem_ids = []
        for i in tqdm(range(len(ds_slice)), desc=f"filtering {i}-{min(i+64000, num_samples)}"):
            try:
                sample = ds_slice[i]
            except:
                problem_ids.append(i)
        ds_slice = ds_slice.select(
            [i for i in range(len(ds_slice)) if i not in problem_ids])
        ds_slice.save_to_disk(f"{split}_v1/{batch}", num_proc=4)
        print(f"complete saving {split}_v1/{batch}", flush=True)

    slices = get_all_split(f"{split}_v1")
    ds     = concatenate_datasets([load_from_disk(split) for split in slices])
    ds.save_to_disk(f"{os.path.dirname(split)}_v1/{os.path.basename(split)}", num_proc=4)


def main(dir):
    splits = get_all_split(dir)
    for split in splits:
        print("start {}".format(split), flush=True)
        map2schema(split)
        print("complete {}".format(split), flush=True)


if __name__ == "__main__":
    Fire(main)
