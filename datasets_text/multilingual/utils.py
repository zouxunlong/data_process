import json
import os
import re
from datasets import Dataset, load_from_disk, concatenate_datasets
from simhash import Simhash
from collections import defaultdict, Counter


pattern_punctuation = r"""[ \n\\!?,*.�:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""


def combine_jsonl(dir):
    files = os.listdir(dir)
    lines = []
    for i, file in enumerate(files):
        lines.extend(open(dir+"/"+file).readlines())
        print(i, flush=True)
    open(dir+".jsonl", "w", encoding="utf8").write("".join(lines))


def combine_dataset(dataset_paths):

    def check_uniques(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    print("start", flush=True)
    ds = concatenate_datasets([load_from_disk(dataset_path) for dataset_path in dataset_paths]).shuffle(seed=42)
    # uniques = set(ds.unique("simhash"))
    # ds = ds.filter(check_uniques, num_proc=4, load_from_cache_file=True, keep_in_memory=False,  fn_kwargs={"uniques": uniques},)
    dir = os.path.dirname(dataset_paths[0])
    docs = len(ds["length"])
    words = sum(ds["length"])
    language = "all"
    version = "combined"
    ds.save_to_disk(dir + "/{}.{}.{}.{}.hf".format(language, version, docs, words), num_proc=10)
    print("completed", flush=True)


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


def map_attributes(dataset_path):

    def mapping(example):
        text = example["text"]
        text = re.sub(r' +', " ", text)
        return {"text": text, "language": "id", "source": "id_seamc", "tag": ["2024/2/29"]}

    dataset = Dataset.from_json(dataset_path)
    columns = dataset.column_names
    columns.remove("text")
    dataset_removed = dataset.remove_columns(column_names=columns)
    dataset_mapped = dataset_removed.map(mapping, num_proc=20)
    dataset_mapped.to_json("{}".format(dataset_path), force_ascii=False)
    dataset_mapped.save_to_disk("{}.hf".format(dataset_path))
    print("completed", flush=True)


def map_length(dataset_path):

    def add_length(batch):
        length = [len(batch["text"][i].split()) if batch["language"][i] in [
            "en", "id"] else len(batch["text"][i]) for i in range(len(batch["text"]))]
        return {"length": length}

    ds = Dataset.load_from_disk(dataset_path)
    ds = ds.map(add_length, batched=True, num_proc=25)
    docs = len(ds["length"])
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    dir = os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.jsonl.hf".format(source, docs, words))
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


def deduplicate(dataset_path):

    def check_uniques(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    print("start {}...".format(dataset_path))
    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    uniques = set(ds.unique("simhash"))
    ds = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
    dir = os.path.dirname(dataset_path)
    docs = len(ds["length"])
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    ds.save_to_disk(dir + "/{}.{}.{}.de.jsonl.hf".format(source, docs, words))
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
    dir = os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.jsonl.hf".format(source, docs, words))
    print("completed {}".format(dataset_path), flush=True)


def select(dataset_path, num):

    ds = load_from_disk(dataset_path)
    docs = len(ds)
    print(ds.column_names, flush=True)
    # ds=ds.select(sorted(random.sample(range(docs), num)))
    ds = ds.select([i for i in range(num)])
    print(ds.unique("source"), flush=True)
    docs = len(ds)
    words = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir = os.path.dirname(dataset_path)
    ds.save_to_disk(
        dir + "/{}.{}.{}.{}.hf".format(language, source, docs, words))


def map2doc(dataset_path):

    def build_doc(batch):
        text = ""
        length = 0
        new_batch = defaultdict(list)
        for i in range(len(batch["length"])):
            if batch["length"][i] > 3900:
                new_batch["text"].append(batch["text"][i])
                new_batch["language"].append(batch["language"][i])
                new_batch["source"].append(batch["source"][i])
                new_batch["length"].append(batch["length"][i])
            else:
                length += batch["length"][i]
                text += batch["text"][i]+"\n\n"
                if length > 3900:
                    new_batch["text"].append(text)
                    new_batch["language"].append(batch["language"][i])
                    new_batch["source"].append(batch["source"][i])
                    new_batch["length"].append(length)
                    text = ""
                    length = 0
        new_batch["text"].append(text)
        new_batch["language"].append(batch["language"][i])
        new_batch["source"].append(batch["source"][i])
        new_batch["length"].append(length)
        return new_batch

    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    print(ds.unique("source"), flush=True)

    ds = ds.map(build_doc, batched=True,
                remove_columns=ds.column_names, num_proc=24)

    print(ds.column_names, flush=True)
    print(ds.unique("source"), flush=True)

    docs = len(ds["length"])
    words = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir = os.path.dirname(dataset_path)
    ds.save_to_disk(
        dir + "/{}.{}.{}.{}.hf".format(language, source, docs, words))
    print("completed", flush=True)


def filter_short(dataset_path):

    def short(batch):
        return [True if length > 10 else False for length in batch["length"]]

    print("start {}...".format(dataset_path))
    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    ds = ds.filter(short, batched=True, num_proc=10)
    dir = os.path.dirname(dataset_path)
    docs = len(ds["length"])
    words = sum(ds["length"])
    source = dataset_path.split("/")[-1].split(".")[0]
    ds.save_to_disk(
        dir + "/{}.{}.{}.hf".format(source, docs, words), num_proc=4)
    print("completed {}".format(dataset_path), flush=True)


def map2tokens(dataset_path):

    from transformers import AutoTokenizer

    TOKENIZER_PATH = './tokenizers'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

    def tokenize(example):
        input_ids = tokenizer.encode(example["text"])
        length = len(input_ids)
        return {"input_ids": input_ids, "length": length}

    ds = load_from_disk(dataset_path)
    ds = ds.map(tokenize, remove_columns=["text", "tag", "simhash"], num_proc=20)
    dir = os.path.dirname(dataset_path)
    docs = len(ds["length"])
    tokens = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    ds.save_to_disk(dir + "/{}.{}.{}.{}.ids.hf".format(language, source, docs, tokens), num_proc=10)
    print("completed", flush=True)


def map2string(dataset_path):

    from transformers import AutoTokenizer

    TOKENIZER_PATH = './tokenizers'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

    def de_tokenize(batch):
        text = [tokenizer.decode(input_ids)
                for input_ids in batch["input_ids"]]
        return {"text": text}

    ds = load_from_disk(dataset_path)

    ds = ds.map(de_tokenize, batched=True, batch_size=100,
                remove_columns=["input_ids"], num_proc=28)

    # docs = len(ds["length"])
    # tokens = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir = os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.text.hf".format(language, source))
    print("completed", flush=True)


def chunk(dataset_path):

    def chunking(batch):

        tmp_input_ids = []
        new_batch = defaultdict(list)
        for i in range(len(batch["input_ids"])):
            id_list = batch["input_ids"][i]
            chunk_size = 8190
            while len(id_list) > chunk_size:
                chunk, id_list = id_list[:chunk_size], id_list[chunk_size:]
                new_batch["input_ids"].append([1]+chunk+[2])
                new_batch["language"].append(batch["language"][i])
                new_batch["source"].append(batch["source"][i])
            if len(id_list) < 20:
                continue

            tmp_input_ids += [1]+id_list+[2]

            if len(tmp_input_ids) > chunk_size+1:
                chunk, tmp_input_ids = tmp_input_ids[:chunk_size +
                                                     1], tmp_input_ids[chunk_size+1:]
                new_batch["input_ids"].append(chunk+[2])
                new_batch["language"].append(batch["language"][i])
                new_batch["source"].append(batch["source"][i])

            if len(tmp_input_ids) < 20:
                tmp_input_ids.clear()
            elif not tmp_input_ids[0] == 1:
                tmp_input_ids = [1]+tmp_input_ids

        return new_batch

    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    ds = ds.map(chunking, batched=True, batch_size=10000,
                remove_columns=ds.column_names, num_proc=16)
    docs = len(ds)
    tokens = docs*8192
    language = dataset_path.split("/")[-1].split(".")[0]
    source = dataset_path.split("/")[-1].split(".")[1]
    dir = os.path.dirname(dataset_path)
    ds.save_to_disk(dir + "/{}.{}.{}.{}.ids.hf".format(language,
                    source, docs, tokens), num_proc=16)
    print("completed", flush=True)


def statistics(dataset_path):
    ds = load_from_disk(dataset_path)
    c = Counter(ds["source"])
    print(c.most_common(), flush=True)


def filter_hardwarezone(dataset_path):

    def remove(batch):
        return [False if source == "en_hardawrezone" else True for source in batch["source"]]

    print("start {}...".format(dataset_path))
    ds = load_from_disk(dataset_path)
    print(ds.column_names, flush=True)
    print(len(ds), flush=True)
    ds = ds.filter(remove, batched=True,
                   load_from_cache_file=True, keep_in_memory=False)
    print(ds.column_names, flush=True)
    print(len(ds), flush=True)
    dir = os.path.dirname(dataset_path)
    docs = len(ds["length"])
    words = sum(ds["length"])
    language = dataset_path.split("/")[-1].split(".")[0]
    version = dataset_path.split("/")[-1].split(".")[1]
    ds.save_to_disk(dir + "/{}.{}.{}.{}.hf".format(language,
                    version, docs, words), num_proc=4)
    print("completed {}".format(dataset_path), flush=True)


if __name__ == "__main__":

    import time
    start = time.time()
    print(os.getpid(), flush=True)

    chunk("/home/user/data/data_text/multilingual/en.combined.10178012.9548925266.ids.hf")

    print("------take {} seconds---".format(time.time()-start), flush=True)


