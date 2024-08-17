import json
from datasets import load_dataset
import os


def download_text():

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
    dataset.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_wikipedia/wikipedia_20231101_en")
    
    dataset = load_dataset("wikimedia/wikipedia", "20231101.zh")
    dataset.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_wikipedia/wikipedia_20231101_zh")



if __name__ == "__main__":
    print(os.getpid(), flush=True)
    download_text()
    print("complet all", flush=True)

