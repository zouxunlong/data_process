import unicodedata
from datasets import load_from_disk, Audio, Features, Value
import random
import re
import string

instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]

translator = str.maketrans('', '', string.punctuation)

def normalize_sentence_for_filter(sentence):
    sentence = unicodedata.normalize('NFKC', sentence.translate(translator))
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
    sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def normalize_sentence(sentence):
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
    sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence

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
            "text": "<Speaker1>: " + normalize_sentence(example["transcription"]),
            "audio": None
        },
        "other_attributes": {
            "id"       : example["id"],
            "speaker"  : example["speaker"],
            "channel"  : example["channel"],
            "session"  : example["session"],
            "partition": example["partition"]
        }
    }


def map2schema(ds, workers=112):

    features = Features({
        'context'         : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction'     : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer'          : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "id"       : ds.features["id"],
            "speaker"  : ds.features["speaker"],
            "channel"  : ds.features["channel"],
            "session"  : ds.features["session"],
            "partition": ds.features["partition"],
        }
    })

    ds = ds.map(mapping,
                features          = features,
                remove_columns    = ds.column_names,
                num_proc          = workers,
                batch_size        = 1,
                writer_batch_size = 1
                )
    return ds


def build(split, workers=56):

    print("start {}".format(split), flush=True)
    part = split.split("/")[-1]
    ds   = load_from_disk(split)
    print(ds, flush=True)

    ds_test = load_from_disk(f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/ASR/IMDA_{part}_ASR_v4")
    test_transcriptions=set()
    for sample in ds_test["answer"]:
        test_transcriptions.add(normalize_sentence_for_filter(sample["text"].replace("<Speaker1>: ", "")).lower().strip())
    print(len(test_transcriptions), flush=True)
    print(next(iter(test_transcriptions)), flush=True)

    print(len(ds), flush=True)
    ds_train = ds.filter(lambda x: [normalize_sentence_for_filter(transcription).lower().strip() not in test_transcriptions for transcription in x["transcription"]], 
                            batched=True, batch_size=1000, writer_batch_size=1000, num_proc=workers)
    print(len(ds_train), flush=True)

    ds_train = map2schema(ds_train)
    ds_train.save_to_disk(f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/train/ASR/IMDA_{part}_ASR", num_proc=10)


def main(splits=[
    "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART1_bytes",
    "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART2_bytes"
]):
    for split in splits:
        build(split)
    print("complete all", flush=True)


if __name__ == "__main__":
    main()
