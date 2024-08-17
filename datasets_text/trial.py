import traceback
from transformers import AutoTokenizer
import os
from datasets import load_from_disk

TOKENIZER_PATH = './tokenizer_llama3_70B'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

def tokenize_batch(batch):
    try:
        tokens=tokenizer(batch["text"], add_special_tokens=False, return_length=True, return_tensors="np")["length"]
        return {"tokens": tokens}
    except Exception as e:
        print(traceback.format_exc(), flush=True)

split="/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/local.new/zh/news.local.zh.hf"
print("start {}".format(split), flush=True)
ds=load_from_disk(split)
ds=ds.map(tokenize_batch, batched=True, batch_size=20, num_proc=5)
ds.save_to_disk(split.replace("local.new/", "local.new2/"), num_proc=5)
print("complete {}".format(split), flush=True)

