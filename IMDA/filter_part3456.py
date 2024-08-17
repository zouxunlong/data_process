import os
import traceback
from datasets import load_from_disk
import re


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def normalize_sentence(sentence):
    return " ".join(re.sub('\s?(\[|\(|<)[a-zA-Z0-9]*(\]|\)|>)\s?', " ", re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)).split()).strip()


def filtering(batch):
    return [bool(normalize_sentence(answer["text"])) for answer in batch["answer"]]


def mapping(batch):
    batch["answer"]=[{"text": normalize_sentence(answer["text"]), "audio": None} for answer in batch["answer"] ]
    return batch


def main(*args, workers=28):
    for part in args:
        splits=get_all_split("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/IMDA.ASR.schemed/{}.ASR.schemed".format(part))
        for split in splits:
            ds=load_from_disk(split)
            print("start {}".format(split), flush=True)
            ds=ds.filter(filtering, num_proc=workers, batched=True, batch_size=100, writer_batch_size=100)
            # ds=ds.map(mapping, num_proc=workers, batched=True, batch_size=100, writer_batch_size=100)
            ds.save_to_disk(split.replace("IMDA.ASR.schemed/","IMDA.ASR.filtered.schemed/"), num_proc=4)
            print("complete {}".format(split), flush=True)


if __name__=="__main__":
    from fire import Fire
    Fire(main)
