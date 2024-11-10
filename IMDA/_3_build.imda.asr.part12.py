from collections import defaultdict
from datasets import load_from_disk, Audio, Features, Value
import random
import re


instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


def normalize_sentence(sentence):
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('(^|\s)<[a-zA-Z0-9]*>($|\s)', " ", sentence)
    sentence = re.sub('(^|\s)(\(ppb\)|\(ppc\)|\(ppl\)|\(ppo\))($|\s)', " ", sentence)
    sentence = re.sub('(^|\s)<[A-Z0-9]*/>($|\s)', " ", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def map2schema(ds, workers=112):

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
                "text": normalize_sentence(example["transcription"]),
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


def split_test(split):

    print("start {}".format(split), flush=True)
    partition = split.split("/")[-1]
    ds        = load_from_disk(split)

    transcriptions          = ds.unique("transcription")
    selected_transcriptions = random.sample(transcriptions, 1000)

    ds_test=ds.filter(lambda x: [transcription in selected_transcriptions for transcription in x["transcription"]], batched=True, batch_size=1000, writer_batch_size=1000, num_proc=112)
    num_dict=defaultdict(int)
    ids2remove=[]
    for i, trans in enumerate(ds_test["transcription"]):
        if num_dict[trans]<3:
            num_dict[trans]+=1
        else:
            ids2remove.append(i)
    ds_test=ds_test.select([i for i in range(len(ds_test)) if i not in ids2remove])
    ds_test=map2schema(ds_test)
    ds_test.save_to_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF_ASR/test/{partition}")

    ds_train=ds.filter(lambda x: [transcription not in selected_transcriptions for transcription in x["transcription"]], batched=True, batch_size=1000, writer_batch_size=1000, num_proc=112)
    ds_train=map2schema(ds_train)
    ds_train.save_to_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF_ASR/train/{partition}")


def main(splits=[
    "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART1",
    "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART2"
]):
    for split in splits:
        split_test(split)
    print("complete all", flush=True)


if __name__ == "__main__":
    main()
