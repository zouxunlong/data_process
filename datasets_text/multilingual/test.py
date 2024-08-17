import json
import multiprocessing as mp
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import Dataset, load_from_disk, concatenate_datasets

# **Configuration**
TOKENIZER_PATH = './tokenizers'

# **Load the tokenizer**
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)  

# **Tokenization function**
def tokenize_batch(examples):
    return tokenizer(examples, padding=True, return_tensors="pt")


def process(item):
    text = item["text"]
    eos=tokenizer.eos_token
    eos_id=tokenizer.eos_token_id
    bos=tokenizer.bos_token
    bos_id=tokenizer.bos_token_id
    tokenized_res_0 = tokenizer.encode(text)
    tokenized_res = tokenizer.encode(text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text+text)
    text_new_0=tokenizer.decode(tokenized_res_0)
    text_new=tokenizer.decode(tokenized_res)
    print(len(tokenized_res_0),flush=True)
    print(len(tokenized_res),flush=True)
    print(len(text.split()),flush=True)
    print(len(text_new_0.split()),flush=True)
    print(len(text_new.split()),flush=True)


ds=load_from_disk("/home/user/data/data_text/v4/en/en.wikipedia.16946.29571094.hf")

for item in ds:
    items=process(item)
    print(items)

