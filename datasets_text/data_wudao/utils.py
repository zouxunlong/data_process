from glob import glob
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


def json2jsonl(file_json, file_jsonl):
    dataset = Dataset.from_json(file_json)
    dataset.to_json(file_jsonl, force_ascii=False)
    print("complete {}".format(file_json), flush=True)


def jsonl2hf():
    jsonl_files=glob("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_wudao/WuDaoCorpus2.0_200G/*.jsonl")
    jsonl_files.sort()
    dataset = Dataset.from_json(jsonl_files, num_proc=40)
    dataset.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_wudao/WuDaoCorpus2_200G_zh", num_proc=40)
    print("complete.", flush=True)


if __name__ == "__main__":
    jsonl2hf()

