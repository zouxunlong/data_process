import json
import multiprocessing as mp
import os
from transformers import AutoTokenizer
from tqdm import tqdm

# **Configuration**
MODEL_NAME = '/scratch/project_462000514/hx/models/sea_mistral_7b'
TOKENIZER_PATH = '/home/user/data/data_text/tokenizers'
JSONL_FILE = "/scratch/project_462000514/hx/pretrain_data/multilingual.train.jsonl"  
OUTPUT_FILE = '/scratch/project_462000514/hx/pretrain_data/multilingual.8k.train.jsonl'
TEXT_KEY = "text"  # Key where the text is stored in your JSON objects
NUM_PROCESSES = 56
TOKEN_SIZE=8192
MINIMUM_TOKEN=8392
MIN_TOKEN_PER_ROW=500
BUFFER_SIZE=100000

# **Load the tokenizer**
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)  

# **Tokenization function**
def tokenize_batch(examples):
    return tokenizer(examples, truncation=True, padding=True, return_tensors="pt")

# **Process a chunk of JSONL data**
def process_chunk(line):
    line_json = json.loads(line)
    text = line_json[TEXT_KEY]
    tokenized_res = tokenizer(text).input_ids[1:]
    if len(tokenized_res) < MINIMUM_TOKEN:
        return [line_json]
    tokenized_res_grouped = [tokenized_res[i: min(i + TOKEN_SIZE, len(tokenized_res))] for i in range(0, len(tokenized_res), TOKEN_SIZE)]
    if len(tokenized_res_grouped[-1]) < MIN_TOKEN_PER_ROW:
        tokenized_res_grouped = tokenized_res_grouped[:-1]
    tokenized_res_reverted = [tokenizer.decode(e) for e in tokenized_res_grouped]
    language = line_json['language']
    source = line_json['source']
    return [{'text': e, 'language': language, 'source': source} for e in tokenized_res_reverted]


def process_lines(f_write, lines):
    tokenized_corpus = []
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = [pool.apply_async(process_chunk, args=(line,)) for line in lines]

        for result in tqdm(results):
            tokenized_corpus += result.get()
    
    print('row before:', len(lines))
    print('row after:', len(tokenized_corpus))
    for l in tokenized_corpus:
        f_write.writelines(json.dumps(l, ensure_ascii=False) + '\n')

# **Main multiprocessing logic**
if __name__ == "__main__":
    lines = []
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_write:
        with open(JSONL_FILE, 'r', encoding='utf-8') as f:
            for l in tqdm(f):
                lines.append(l)
                if len(lines) >= BUFFER_SIZE:
                    process_lines(f_write, lines)
                    lines = []
    if len(lines) >= 0:
        process_lines(f_write, lines)
