import random
import traceback
import os
import re
from datasets import load_from_disk, Features, Value, Audio
import fire
import unicodedata


instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


def normalize_sentence(sentence):
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
    sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def normalize_transcription(transcription, audio_length, max_length):

    normalized_transcription = []
    tmp_transcription = {}
    for utterance in transcription:

        sentence = normalize_sentence(utterance["sentence"])
        if not sentence or utterance["start"] < 0 or utterance["end"] > audio_length:
            continue
        if tmp_transcription == {}:
            if utterance["end"] - utterance["start"] < max_length:
                tmp_transcription = {
                    "start"   : utterance["start"],
                    "end"     : utterance["end"],
                    "sentence": sentence
                    }
        elif utterance["start"] == tmp_transcription["end"] and utterance["end"] - tmp_transcription["start"] < max_length:
            tmp_transcription["sentence"] += " " + sentence
            tmp_transcription["end"] = utterance["end"]
        else:
            normalized_transcription.append({
                "start"   : tmp_transcription["start"],
                "end"     : tmp_transcription["end"],
                "sentence": tmp_transcription["sentence"]
                })
            tmp_transcription.clear()
            if utterance["end"] - utterance["start"] < max_length:
                tmp_transcription = {
                    "start"   : utterance["start"],
                    "end"     : utterance["end"],
                    "sentence": sentence
                    }

    return normalized_transcription


def map_fn(batch, max_length):
    try:
        new_batch = {
            "context"         : [],
            "instruction"     : [],
            "answer"          : [],
            "other_attributes": []
        }

        array                    = batch["audio"][0]["array"]
        audio_length             = array.size/16000
        transcription            = batch["transcription"][0]
        normalized_transcription = normalize_transcription(transcription, audio_length, max_length)

        for utterance in normalized_transcription:
            start_time  = utterance["start"]
            end_time    = utterance["end"]
            chunk_array = array[int(start_time*16000):int(end_time*16000)]

            new_batch["context"].append({
                "text": None,
                "audio": {"array": chunk_array, "sampling_rate": 16000}
            })
            new_batch["instruction"].append({
                "text": random.choice(instructions_asr),
                "audio": None
            })
            new_batch["answer"].append({
                "text": "<Speaker1>: " + utterance["sentence"],
                "audio": None
            })
            new_batch["other_attributes"].append({
                "conversation_id": batch["conversation_id"][0],
                "speaker"        : batch["speaker"][0],
                "setting"        : batch["setting"][0],
                "partition"      : batch["partition"][0],
                "start"          : start_time,
                "end"            : end_time
            })
        return new_batch
    except:
        print(traceback.format_exc(), flush=True)


def map2schema(ds, max_length, workers=56):
    features = Features({
        'context'         : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction'     : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer'          : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "conversation_id": ds.features["conversation_id"],
            "speaker"        : ds.features["speaker"],
            "setting"        : ds.features["setting"],
            "partition"      : ds.features["partition"],
            "start"          : Value(dtype="float64"),
            "end"            : Value(dtype="float64"),
        }
    })

    ds = ds.map(map_fn,
                fn_kwargs         = {"max_length": max_length},
                batched           = True,
                batch_size        = 1,
                writer_batch_size = 1,
                features          = features,
                remove_columns    = ds.column_names,
                num_proc          = workers)
    return ds


def build_asr(split, max_length, workers=56):

    part = split.split("/")[-1]

    ds = load_from_disk(split)
    ds = map2schema(ds, max_length)

    ds_test = load_from_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/test/ASR/IMDA_{part}_ASR_v4")
    test_transcriptions=set()
    for sample in ds_test:
        test_transcriptions.add(normalize_sentence(sample["answer"]["text"]).lower())
    print("samples of test_transcriptions: ", len(test_transcriptions), flush=True)

    print(len(ds), flush=True)
    ds_train    = ds.filter(lambda x: [normalize_sentence(answer["text"].replace("<Speaker1>: ", "")).lower() not in test_transcriptions for answer in x["answer"]], 
                            batched=True, batch_size=1000, writer_batch_size=1000, num_proc=workers)
    print(len(ds_train), flush=True)

    ds_train.save_to_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_ASR_v4", num_proc=10)


def main():
    splits=[
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART3",
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART4",
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART5",
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_mono_hf/PART6",
    ]
    for split in splits:
        print("start {}".format(split), flush=True)
        build_asr(split=split, max_length=30)
    print("complete all", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
