
from datasets import load_from_disk, Audio, Features, Value
import random
from fire import Fire


instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


def map2schema(ds_path, workers=32):

    def mapping(example):
        return {
            "context": {
                "text": None,
                "audio": example["audio"]
            },
            "instruction": {
                "text": random.choice(instructions_asr),
                "audio": None
            },
            "answer": {
                "text": example["sentence"],
                "audio": None
            },
            "other_attributes": {
                "up_votes"  : example["up_votes"],
                "down_votes": example["down_votes"],
                "age"       : example["age"],
                "gender"    : example["gender"],
                "accent"    : example["accent"],
                "locale"    : example["locale"],
                "segment"   : example["segment"],
                "variant"   : example["variant"],
            }
        }

    ds = load_from_disk(ds_path)
    features = Features({
        'context'         : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction'     : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer'          : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "up_votes"  : ds["test"].features["up_votes"],
            "down_votes": ds["test"].features["down_votes"],
            "age"       : ds["test"].features["age"],
            "gender"    : ds["test"].features["gender"],
            "accent"    : ds["test"].features["accent"],
            "locale"    : ds["test"].features["locale"],
            "segment"   : ds["test"].features["segment"],
            "variant"   : ds["test"].features["variant"],
        }
    })

    ds_dict = ds.map(mapping,
                     features          = features,
                     remove_columns    = ds["test"].column_names,
                     num_proc          = workers,
                     batch_size        = 1,
                     writer_batch_size = 1
                     )

    ds_dict.save_to_disk(ds_path+"_v1", num_proc=4)
    print(f"complete {ds_path}", flush=True)


if __name__ == "__main__":
    Fire(map2schema)
