import json
import os
from datasets import Dataset


def download_text():

    import subprocess
    command = 'wget https://download.scidb.cn/download?fileId=63a30383fed6a8a9e8454302&dataSetType=organization&fileName=WuDaoCorporaText-2.0-open.rar'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("output: {}".format(output), flush=True)
    print("error: {}".format(error), flush=True)


def count_tokens_file(file):
    count=0
    with open(file) as f:
        for i, line in enumerate(f):
            if i%10000==0:
                print(i, flush=True)
            item=json.loads(line)
            content = item["text"]
            count+=len(content)
    print("{} line: {} tokens.".format(i, count), flush=True)
    print("complete", flush=True)


def json2jsonl(file_json, file_jsonl):
    dataset = Dataset.from_json(file_json)
    dataset.to_json(file_jsonl, force_ascii=False)
    print("complete {}".format(file_json), flush=True)


if __name__ == "__main__":
    dir="/home/user/data/data_text/data_wudao/WuDaoCorpus2.0_base_200G"
    dir_jsonl="/home/user/data/data_text/data_wudao/wudao"
    files=os.listdir(dir)
    files.sort()
    for file in files:
        file_path_json=os.path.join(dir, file)
        file_path_jsonl=os.path.join(dir_jsonl, file+"l")
        json2jsonl(file_path_json, file_path_jsonl)

