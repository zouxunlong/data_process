import json
import urllib.request
import os

def download_audio():
    with open("english_listencing_comprehension/hxen_json/hxen.jsonl") as f_in:
        for line in f_in:
            item=json.loads(line)
            audio_link=item["audio_link"]
            title=item["title"]
            print(audio_link,flush=True)
            urllib.request.urlretrieve(audio_link, "english_listencing_comprehension/hxen_audio/{}.mp3".format(title))


def extract_text():
    with open("/home/user/data/data_SQA/english_listencing_comprehension/hxen_json/hxen.jsonl") as f_in:
        for line in f_in:
            item=json.loads(line)
            text=item["text"]
            title=item["title"]
            open("/home/user/data/data_SQA/english_listencing_comprehension/hxen_audio/{}.txt".format(title),"w", encoding="utf8").write(text)


def build_samples(file):

    def generate_sample(exam, chunk):
        lines=chunk.strip().split("\n")
        assert len(lines)==4
        sample={}
        sample["id"], sample["question"]=lines[0].split(". ")
        sample["id"] = exam + sample["id"]
        sample["answer"] = "("+chunk[4][0]+")"+" "+ chunk[4][2:].strip()
        sample["choices"]=["("+line[0]+")"+" "+line[2:].strip() for line in chunk[1:4]]
        return sample

    contend=open(file).read()
    exam=file.split("/")[-1].split(".")[0]
    chunks=contend.split("\n\n")
    for chunk in chunks:
        sample=generate_sample(exam, chunk)
        open(os.path.dirname(file)+"/samples.jsonl", "a", encoding="utf8").write(json.dumps(sample, ensure_ascii=False)+"\n")



if __name__=="__main__":
    
    # build_samples()
    from datasets import Dataset, Audio, load_from_disk

    # audio_dataset = Dataset.from_json("/home/user/data/data_SQA/sample.jsonl", split="test").cast_column("audio", Audio())

    audio_dataset= load_from_disk("/home/user/data/data_SQA/sample.hf")
    for item in audio_dataset:
        print(item, flush=True)

