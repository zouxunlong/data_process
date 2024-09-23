import json
import os
import re
from datasets import Dataset, load_from_disk, concatenate_datasets
from simhash import Simhash
from collections import defaultdict, Counter
from tqdm import tqdm

pattern_punctuation = r"""[ \n\\!?,*.�:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""


def combine_jsonl(dir):
    files = os.listdir(dir)
    lines = []
    for i, file in enumerate(files):
        lines.extend(open(dir+"/"+file).readlines())
        print(i, flush=True)
    open(dir+".jsonl", "w", encoding="utf8").write("".join(lines))


def combine_dataset(datasets):

    def check_uniques(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    print("start", flush=True)
    ds=concatenate_datasets([load_from_disk(dataset_path) for dataset_path in datasets])
    # uniques = set(ds.unique("simhash"))
    # ds = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
    docs=len(ds)
    words = sum(ds["length"])
    ds = ds.train_test_split(test_size=0.0001, shuffle=True)
    ds.save_to_disk("/home/user/data/data_text/v3/v3.slimpajama.{}.{}.hf".format(docs, words), num_proc=25)
    print("completed.", flush=True)


def count_words_dir(dir):
    count = 0
    files = os.listdir(dir)
    files.sort()
    for file in files:
        text = open(dir+"/"+file).read()
        count += len(text.split())
        print(count, flush=True)
    print("complete", flush=True)


def count_words_file(file):
    count = 0
    with open(file) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(i, flush=True)
            item = json.loads(line)
            text = item["text"]
            count += len(text.split())
    print("{} line: {} tokens.".format(i+1, count), flush=True)
    print("complete", flush=True)


def count_length_hf(dataset_path):
    ds = load_from_disk(dataset_path)
    lengths = ds["length"]
    total_length = sum(lengths)
    print("dataset {} : {} lines; {} words".format(
        dataset_path, len(lengths), total_length), flush=True)


def countlines(file):
    lines = open(file).readlines()
    print("{}: {} lines".format(file, len(lines)), flush=True)


def json2jsonl(file_json, file_jsonl):
    dataset = Dataset.from_json(file_json)
    dataset.to_json(file_jsonl, force_ascii=False)
    print("complete {}".format(file_json), flush=True)


def map_attributes(dataset_path):

    def mapping(example):
        text = example["text"]
        source = example["source"]
        language=source[:2]
        text = re.sub(r' +', " ", text)
        return {"text": text, "language": language}

    dataset = load_from_disk(dataset_path)
    print(dataset.column_names, flush=True)
    dataset = dataset.map(mapping, remove_columns=["id"], num_proc=20)
    print(dataset.column_names, flush=True)
    dataset.save_to_disk(dataset_path.replace(".hf", ".attr.hf"), num_proc=20)
    print("completed", flush=True)


def map_length(dataset_path):

    def add_length(batch):
        length = [len(batch["text"][i].split()) if batch["language"][i] in [
            "en", "id"] else len(batch["text"][i]) for i in range(len(batch["text"]))]
        return {"length": length}

    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    ds = ds.map(add_length, batched=True, num_proc=25)
    print(ds.column_names, flush=True)
    docs = len(ds["length"])
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.jsonl.hf".format(source, docs, words), num_proc=25)
    print("completed {}".format(dataset_path), flush=True)


def reduce_attribute(dataset_path):
    with open(dataset_path) as f, \
            open(dataset_path+".mapped", "w", encoding="utf8") as f_out:
        for i, line in enumerate(f):
            item = json.loads(line)
            f_out.write(json.dumps({"text": item["text"], "language": "en", "source": "en_news", "tag": [
                        "2024/2/29"]}, ensure_ascii=False)+"\n")
            if i % 100000 == 0:
                print(i, flush=True)
    print("completed {} lines".format(i+1), flush=True)


def jsonl2hf(dataset_path, language, source, tag):

    def map_attribute(example):
        return {"simhash": Simhash(example['text'], f=40).value, 
                "language": language, 
                "source": source,
                "tag": [tag]}

    def check_uniques(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    def add_length(batch):
        length = [len(batch["text"][i].split()) if batch["language"][i] in [
            "en", "id"] else len(batch["text"][i]) for i in range(len(batch["text"]))]
        return {"length": length}

    print("start {}...".format(dataset_path))
    ds = Dataset.from_json(dataset_path)
    print(ds.column_names, flush=True)
    columns=ds.column_names
    columns.remove("text")
    ds = ds.map(map_attribute, num_proc=20, remove_columns=columns)
    print(ds.column_names, flush=True)
    uniques = set(ds.unique("simhash"))
    ds = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
    ds = ds.map(add_length, batched=True, num_proc=20)
    print(ds.column_names, flush=True)
    docs = len(ds["length"])
    words = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    ds.save_to_disk("/home/user/data/data_text/v4/{}" + "/{}.{}.{}.{}.jsonl.hf".format(language, language, source, docs, words))
    print("completed {}".format(dataset_path), flush=True)


def deduplicate(dataset_path):

    def check_uniques(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    uniques = set(ds.unique("simhash"))
    ds = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
    ds=ds.remove_columns(['simhash', 'source', 'tag'])
    print(ds.column_names, flush=True)
    dir=os.path.dirname(dataset_path)
    docs = len(ds)
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    ds.save_to_disk("{}/{}.{}.{}.de.hf".format(dir, source, docs, words), num_proc=20)
    print("completed {}".format(dataset_path), flush=True)


def deduplicate2(dataset_path):

    def add_hash(example):
        return {"simhash": Simhash(example['text'], f=40).value}

    def check_uniques(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    def add_length(batch):
        length = [len(batch["text"][i].split()) if batch["language"][i] in [
            "en", "id"] else len(batch["text"][i]) for i in range(len(batch["text"]))]
        return {"length": length}

    print("start {}...".format(dataset_path))
    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    ds = ds.map(add_hash, num_proc=25)
    print(ds.column_names, flush=True)
    uniques = set(ds.unique("simhash"))
    ds = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
    ds = ds.map(add_length, batched=True, num_proc=25)
    print(ds.column_names, flush=True)
    docs = len(ds["length"])
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.jsonl.hf".format(source, docs, words))
    print("completed {}".format(dataset_path), flush=True)


def select(dataset_path, num):

    ds = load_from_disk(dataset_path)
    docs = len(ds)
    print(ds.column_names, flush=True)
    # ds=ds.select(sorted(random.sample(range(docs), num)))
    ds=ds.select([i for i in range(num)])
    print(ds.unique("source"), flush=True)
    docs = len(ds)
    words = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.{}.hf".format(language, source, docs, words))


def map2doc(dataset_path):

    def build_doc(batch):
        text=""
        length=0
        new_batch=defaultdict(list)
        for i in range(len(batch["length"])):
            if batch["length"][i]>3900:
                new_batch["text"].append(batch["text"][i])
                new_batch["language"].append(batch["language"][i])
                new_batch["source"].append(batch["source"][i])
                new_batch["length"].append(batch["length"][i])
            else:
                length+=batch["length"][i]
                text+=batch["text"][i]+"\n\n"
                if length>3900:
                    new_batch["text"].append(text)
                    new_batch["language"].append(batch["language"][i])
                    new_batch["source"].append(batch["source"][i])
                    new_batch["length"].append(length)
                    text=""
                    length=0
        new_batch["text"].append(text)
        new_batch["language"].append(batch["language"][i])
        new_batch["source"].append(batch["source"][i])
        new_batch["length"].append(length)
        return new_batch


    ds=load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    print(ds.unique("source"), flush=True)

    ds = ds.map(build_doc, batched=True, remove_columns=ds.column_names, num_proc=24)

    print(ds.column_names, flush=True)
    print(ds.unique("source"), flush=True)

    docs = len(ds["length"])
    words = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.{}.hf".format(language, source, docs, words))
    print("completed", flush=True)


def filter_short(dataset_path):

    def short(batch):
        return [True if length>10 else False for length in batch["length"]]


    print("start {}...".format(dataset_path))
    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    ds = ds.filter(short, batched=True, num_proc=10)
    dir=os.path.dirname(dataset_path)
    docs = len(ds["length"])
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    ds.save_to_disk(dir + "/{}.{}.{}.hf".format(source, docs, words), num_proc=4)
    print("completed {}".format(dataset_path), flush=True)


def map2tokens(dataset_path):

    from transformers import AutoTokenizer

    TOKENIZER_PATH = './tokenizers'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

    def tokenize(example):
        input_ids = tokenizer.encode(example["text"])
        length = len(input_ids)
        return {"input_ids": input_ids, "length": length}

    ds=load_from_disk(dataset_path)

    # ds = ds.map(tokenize, batched=True, remove_columns=["text", "tag", "simhash"], num_proc=25)
    ds = ds.map(tokenize, remove_columns=["text", "tag", "simhash"], num_proc=25)

    docs = len(ds["length"])
    tokens = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.{}.hf".format(language, source, docs, tokens))
    print("completed", flush=True)


def map2string(dataset_path):

    from transformers import AutoTokenizer

    TOKENIZER_PATH = './tokenizers'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

    def de_tokenize(batch):
        text = [tokenizer.decode(input_ids) for input_ids in batch["input_ids"]]
        return {"text": text}

    ds=load_from_disk(dataset_path)

    ds = ds.map(de_tokenize, batched=True, batch_size=100, remove_columns=["input_ids"], num_proc=28)

    # docs = len(ds["length"])
    # tokens = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.text.hf".format(language, source))
    print("completed", flush=True)


def chunk(dataset_path):

    def chunking(batch):

        tmp_input_ids=[]
        new_batch=defaultdict(list)
        for i in range(len(batch["input_ids"])):
            id_list=batch["input_ids"][i]
            chunk_size = 8190
            while len(id_list)>chunk_size: 
                chunk, id_list = id_list[:chunk_size], id_list[chunk_size:] 
                new_batch["input_ids"].append([1]+chunk+[2])
                new_batch["language"].append(batch["language"][i])
                new_batch["source"].append(batch["source"][i])
            if len(id_list)<20:
                continue

            tmp_input_ids+=[1]+id_list+[2]

            if len(tmp_input_ids)>chunk_size+1:
                chunk, tmp_input_ids = tmp_input_ids[:chunk_size+1], tmp_input_ids[chunk_size+1:] 
                new_batch["input_ids"].append(chunk+[2])
                new_batch["language"].append(batch["language"][i])
                new_batch["source"].append(batch["source"][i])

            if len(tmp_input_ids)<20:
                tmp_input_ids.clear()
            elif not tmp_input_ids[0]==1:
                tmp_input_ids=[1]+tmp_input_ids

        return new_batch


    ds=load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    ds = ds.map(chunking, batched=True, batch_size=10000, remove_columns=ds.column_names, num_proc=28)
    docs = len(ds)
    tokens = docs*8192
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir=os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.{}.hf".format(language, source, docs, tokens), num_proc=20)
    print("completed", flush=True)


def statistics(dataset_path):
    ds=load_from_disk(dataset_path)
    c=Counter(ds["source"])
    print(c.most_common(), flush=True)


if __name__ == "__main__":


    with open("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/data_text/v3/v3.jsonl") as f_in:
        for line in f_in:
            print(line, flush=True)
            break
