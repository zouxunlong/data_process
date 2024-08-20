import json, os

def examining(file):
    with open(file) as f_in:
        for line in f_in:
            print(json.loads(line), flush=True)
            continue


def json2jsonl(file):
    with open(file) as f_in, open(file.replace(".json", ".jsonl"), "w", encoding="utf8") as f_out:
        for line in f_in:
            item=json.loads(line)
            data=item["data"]
            for i, doc in enumerate(data):
                doc["id"]=doc["id"].replace(".wav", "")
                doc["audio"]="/home/user/data/aqa_data/WavCaps/audio_files/SoundBible/{}.flac".format(doc["id"])
                doc["audio_path"]=doc["audio"]
                new_line=json.dumps({"caption": doc["caption"], "audio":doc["audio"], "audio_path":doc["audio"], "duration":doc["duration"]})+"\n"
                f_out.write(new_line)
            print("total {} docs.".format(i), flush=True)


def combine_srcs(dir):
    lines=[]
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            lines.extend(open(os.path.join(parent_dir, file)).readlines())
    open("/home/user/data/aqa_data/WavCaps/json_files/WavCaps.jsonl","w", encoding="utf8").writelines(lines)


def shuffle(file):
    lines=open(file).readlines()
    import random
    random.shuffle(lines)
    open(file.replace(".jsonl", ".shuffled.jsonl"), "w", encoding="utf8").writelines(lines)


def select_less_than_30s():
    with open("/home/user/data/aqa_data/WavCaps/json_files/WavCaps.test.jsonl") as f_in,\
        open("/home/user/data/aqa_data/WavCaps/json_files/WavCaps.test.less_than_30s.jsonl", "w", encoding="utf8") as f_out:
        for line in f_in:
            item=json.loads(line)
            if item["duration"]<30:
                f_out.write(line)



def split(file):
    lines=open(file).readlines()
    open(file.replace(".jsonl", ".test.jsonl"), "w", encoding="utf8").writelines(lines[:2500])
    open(file.replace(".jsonl", ".validation.jsonl"), "w", encoding="utf8").writelines(lines[2500:5000])
    open(file.replace(".jsonl", ".train.jsonl"), "w", encoding="utf8").writelines(lines[5000:])

def convert2hf():
    from datasets import Dataset, Audio
    audio_dataset = Dataset.from_json("/home/user/data/aqa_data/WavCaps/json_files/WavCaps.train.jsonl", split="train").cast_column("audio", Audio())
    audio_dataset.save_to_disk("/home/user/data/aqa_data/WavCaps/train")


if __name__=="__main__":
    from datasets import load_from_disk

    dataset = load_from_disk("/home/user/data/data_AQA/WavCaps_hf/test")
    for item in dataset:
        print(item, flush=True)
        # break