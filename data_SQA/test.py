
from datasets import load_from_disk
from pprint import pprint

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
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))
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
    print(ds.column_names, flush=True)
    return ds



ds=load_from_disk("/home/all_datasets/datasets_multimodal/ASR/IMDA.ASR_v1/PART1.ASR.schemed/test").select(range(10))

pprint(ds.features)
