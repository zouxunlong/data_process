import os
import re
import traceback
from fire import Fire
from transformers import AutoTokenizer
from simhash import Simhash
from datasets import Dataset, load_from_disk

TOKENIZER_PATH = './tokenizer_llama3_70B'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort()
    return directories


def select(parent_dir):
    try:
        ds=load_from_disk(parent_dir)
        columns=ds.column_names
        if "content" in columns:
            print(parent_dir, flush=True)
            ds=ds.rename_column("content", "text")
        ds=ds.select_columns(["text", "date"])
        ds.save_to_disk(parent_dir.replace("/local/", "/local.text/"), num_proc=5)
        print(ds.column_names, flush=True)
    except Exception as e:
        print(traceback.format_exc(), flush=True)


def add_tokens_simhash(batch):
    try:
        tokens=tokenizer(batch["text"], add_special_tokens=False, return_length=True, return_tensors="np")["length"]
        simhashes=[Simhash(text, f=32).value for text in batch["text"]]
        return {"simhash": simhashes, "tokens": tokens}
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        return {"simhash": [], "tokens": []}


def standardize_text_ds(split, workers):
    ds=load_from_disk(split).select_columns(["text", "date"])
    ds=ds.map(add_tokens_simhash, batched=True, batch_size=20, num_proc=workers)
    ds.save_to_disk("{}.hf".format(split), num_proc=workers)


def main(path, workers=8):
    splits = get_all_split(path)
    for split in splits:
        if os.path.exists("{}.hf".format(split)):
            continue
        standardize_text_ds(split, workers)
    print("complete all", flush=True)


if __name__=="__main__":
    Fire(main)

