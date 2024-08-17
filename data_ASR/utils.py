import json
from typing import Dict
from datasets import load_from_disk, Audio, Features, Value, ClassLabel
import random
import os
from tqdm import tqdm
import numpy as np

instructions_asr = [
    'Please transcribe the speech.',
    'Kindly jot down the speech.',
    'Record the conversation verbatim, please.',
    'Could you write out what was said?',
    'Capture the words spoken accurately.',
    'Transcribe the speech for reference.',
    'Could you document the spoken words precisely?',
    'Please write down the spoken content.',
    'Please provide a written version of the speech.',
    'Could you transcribe the speech accurately.',
    'Create a written record of the spoken communication.',
    'Please write out the speech.',
    'Record the spoken words in text form.',
    'Could you put the speech into writing?',
    'Capture the speech in written format, please.',
    'Transcribe the spoken content, if you will.',
    'Write down what was said, please.',
    'Convert the speech into written text.',
    'Document the conversation by writing it down.',
    'Could you transcribe the verbal exchange?',
    'Please convert the spoken words into text.',
    'Create a written rendition of the speech.',
    'Record the verbal communication in textual form.',
    'Could you provide a written account of the speech?',
    'Transcribe the conversation accurately, please.',
    'Write out the words that were spoken.',
    'Please document the spoken language.',
    'Capture the spoken content in writing.',
    'Convert the speech into a written transcript.',
    'Could you transcribe what you hear?',
    'Please put the spoken words into writing.'
]


def map2schema_commonvoice(dataset_path, num_proc):

    def mapping(example):

        return {"context": {"text": None,
                            "audio": {"array": example["audio"]["array"], "sampling_rate": example["audio"]["sampling_rate"]}},
                "instruction": {"text": random.choice(instructions_asr),
                                "audio": None},
                "answer": {"text": example["sentence"],
                           "audio": None},
                "other_attributes": {"client_id": example["client_id"],
                                     "up_votes": example["up_votes"],
                                     "down_votes": example["down_votes"],
                                     "age": example["age"],
                                     "gender": example["gender"],
                                     "accents": example["accents"],
                                     "variant": example["variant"],
                                     "locale": example["locale"],
                                     "segment": example["segment"],
                                     "language": example["language"],
                                     }
                }

    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "client_id": Value(dtype='string'),
            "up_votes": Value(dtype='int64'),
            "down_votes": Value(dtype='int64'),
            "age": Value(dtype='string'),
            "gender": Value(dtype='string'),
            "accents": Value(dtype='string'),
            "variant": Value(dtype='float64'),
            "locale": Value(dtype='string'),
            "language": Value(dtype='string'),
            "segment": Value(dtype='string'),
        }
    })
    ds = ds.map(mapping, features=features,
                remove_columns=ds.column_names, num_proc=num_proc)
    ds.save_to_disk(dataset_path+".new", num_proc=4)
    print(ds.column_names, flush=True)


def map2schema_commonvoice_no_features(dataset_path, num_proc):

    def mapping(example):

        return {"context": {"text": None,
                            "audio": {"array": np.array(example["audio"]["array"]), "sampling_rate": example["audio"]["sampling_rate"]}},
                "instruction": {"text": random.choice(instructions_asr),
                                "audio": None},
                "answer": {"text": example["sentence"],
                           "audio": None},
                "other_attributes": {"client_id": example["client_id"],
                                     "up_votes": example["up_votes"],
                                     "down_votes": example["down_votes"],
                                     "age": example["age"],
                                     "gender": example["gender"],
                                     "accents": example["accents"],
                                     "variant": example["variant"],
                                     "locale": example["locale"],
                                     "language": example["language"],
                                     }
                }

    ds = load_from_disk(dataset_path)
    print(ds.features, flush=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))
    ds = ds.map(mapping, remove_columns=ds.column_names, num_proc=num_proc)
    ds = ds.cast_column('context', {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)})
    ds = ds.cast_column('instruction', {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)})
    ds = ds.cast_column('answer', {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)})
    print(ds.features, flush=True)
    return ds


def map2schema_dream(dataset_path):

    def mapping(example):

        standard = {"context": {"text": None,
                                "audio": {"array": example["audio"]["array"], "sampling_rate": example["audio"]["sampling_rate"]}},
                    "instruction": {"text":  example["question"],
                                    "audio": None},
                    "answer": {"text": example["answer"],
                               "audio": None},
                    "other_attributes": {"id": example["id"],
                                         "dialogue_id": example["dialogue_id"],
                                         "dialogue": example["dialogue"],
                                         "choices": example["choices"],
                                         "mc_answer": example["mc_answer"]
                                         }
                    }
        return standard

    ds = load_from_disk(dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(ds.column_names, flush=True)
    ds = ds.map(mapping, remove_columns=ds.column_names, num_proc=4)
    print(ds.column_names, flush=True)
    return ds


def map2schema_voxceleb(dataset_path):

    def mapping(example):
        standard = {
            "context": {
                "text": None,
                "audio": {"array": example["audio"]["array"], "sampling_rate": example["audio"]["sampling_rate"]}
            },
            "instruction": {
                "text": example["question"],
                "audio": None
            },
            "answer": {
                "text": example["answer"],
                "audio": None
            },
            "other_attributes": {
                "Gender": example["Gender"],
                "Nationality": example["Nationality"],
                "VGGFace1 ID": example["VGGFace1 ID"],
                "VoxCeleb1 ID": example["VoxCeleb1 ID"],
                "index": example["index"]
            }
        }
        return standard

    ds = load_from_disk(dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(ds.column_names, flush=True)
    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio()},
        'instruction': {"text": Value(dtype='string'), "audio": Audio()},
        'answer': {"text": Value(dtype='string'), "audio": Audio()},
        'other_attributes': {
            "Gender": Value(dtype='string'),
            "Nationality": Value(dtype='string'),
            "VGGFace1 ID": Value(dtype='string'),
            "VoxCeleb1 ID": Value(dtype='string'),
            "index": Value(dtype='string')
        }
    })
    ds = ds.map(mapping, features=features,
                remove_columns=ds.column_names, num_proc=4)
    print(ds.column_names, flush=True)
    return ds


def map2schema_giga(dataset_path, num_proc):

    def mapping(example):
        return {
            "context": {
                "text": None,
                "audio": {"array": example["audio"]["array"], "sampling_rate": example["audio"]["sampling_rate"]}
            },
            "instruction": {
                "text": random.choice(instructions_asr),
                "audio": None
            },
            "answer": {
                "text": example["text"],
                "audio": None
            },
            "other_attributes": {
                "segment_id": example["segment_id"],
                "speaker": example["speaker"],
                "begin_time": example["begin_time"],
                "end_time": example["end_time"],
                "audio_id": example["audio_id"],
                "url": example["url"],
                "source": example["source"],
                "category": example["category"],
            }
        }

    ds = load_from_disk(dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))
    print(ds.column_names, flush=True)
    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "segment_id": Value(dtype='string'),
            "speaker": Value(dtype='string'),
            "begin_time": Value(dtype='float32'),
            "end_time": Value(dtype='float32'),
            "audio_id": Value(dtype='string'),
            "url": Value(dtype='string'),
            "source": ClassLabel(names=['audiobook', 'podcast', 'youtube']),
            "category": ClassLabel(names=["People  and  Blogs", "Business", "Nonprofits  and  Activism", "Crime", "History", "Pets  and  Animals", "News and Politics", "Travel and Events", "Kids and Family", "Leisure", "N/A", "Comedy", "News  and  Politics", "Sports", "Arts", "Science  and  Technology", "Autos  and  Vehicles", "Science and Technology", "People and Blogs", "Music", "Society and Culture", "Education", "Howto  and  Style", "Film  and  Animation", "Gaming", "Entertainment", "Travel  and  Events", "Health and Fitness", "audiobook"]),
        }
    })
    ds = ds.map(mapping, features=features,
                remove_columns=ds.column_names, num_proc=num_proc)
    print(ds.column_names, flush=True)
    return ds


def map2schema_people(dataset_path, num_proc):

    def mapping(example):
        return {
            "context": {
                "text": None,
                "audio": {"array": example["audio"]["array"], "sampling_rate": example["audio"]["sampling_rate"]}
            },
            "instruction": {
                "text": random.choice(instructions_asr),
                "audio": None
            },
            "answer": {
                "text": example["text"],
                "audio": None
            },
            "other_attributes": {
                "id": example["id"],
                "duration_ms": example["duration_ms"],
            }
        }

    ds = load_from_disk(dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))
    print(ds.column_names, flush=True)
    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "id": Value(dtype='string'),
            "duration_ms": Value(dtype='int32'),
        }
    })
    ds = ds.map(mapping, features=features,
                remove_columns=ds.column_names, num_proc=num_proc, writer_batch_size=200)
    print(ds.column_names, flush=True)
    return ds


def cast(dataset_path):
    print("start {}".format(dataset_path), flush=True)
    ds=load_from_disk(dataset_path)
    ds=ds.cast_column(
        'context', {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)}
        ).cast_column(
        'instruction', {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)}
        ).cast_column(
        'answer', {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)}
        )
    return ds


def calculate(dataset_path):
    print("start {}".format(dataset_path), flush=True)
    duration=0
    ds=load_from_disk(dataset_path)
    for item in tqdm(ds):
        length=len(item["context"]["audio"]["array"])/16000
        duration+=length
    return duration


def count_samples(dir):
    dict={}
    for parent_dir, dirs, files in os.walk(dir):
        if len(dirs) == 0:
            try:
                ds = load_from_disk(parent_dir)
                print("{} : {}".format(parent_dir, len(ds)), flush=True)
                dict[parent_dir]=len(ds)
            except Exception as e:
                print("ERROR on {}: {}".format(parent_dir, e), flush=True)
                pass
    with open("{}.samples".format(dir), "w") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    print(os.getpid(), flush=True)

    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "client_id": Value(dtype='string'),
            "up_votes": Value(dtype='int64'),
            "down_votes": Value(dtype='int64'),
            "age": Value(dtype='string'),
            "gender": Value(dtype='string'),
            "accents": Value(dtype='string'),
            "variant": Value(dtype='float64'),
            "locale": Value(dtype='string'),
            "language": Value(dtype='string'),
            "segment": Value(dtype='string'),
        }
    })
    # ds=load_from_disk("/home/all_datasets/datasets_multimodal/ASR/common_voice_15_en_v1/invalidated")
    # print(ds.column_names, flush=True)
    # ds.cast(features, batch_size=10, writer_batch_size=10, num_proc=8)
    # print(ds.column_names, flush=True)
    # ds.save_to_disk("/home/all_datasets/datasets_multimodal/ASR/common_voice_15_en_v1/invalidated.new", num_proc=4)
    ds=load_from_disk("/home/all_datasets/datasets_multimodal/ASR/common_voice_15_en_v1/invalidated").select([0])
    for item in ds:
        print(item[""])
    print("complete all", flush=True)

