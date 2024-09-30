import re
from datasets import Dataset
import os
import json


src="ta_Seithi"
lang="ta"

with open("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/ta.jsonl", 'a', encoding='utf8') as fOut:
    for root_dir, dirs, files in os.walk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/ta_Seithi"):
        for file in files:
            with open(os.path.join(root_dir, file), 'r', encoding='utf8') as f:
                for line in f:
                    item = json.loads(line)
                    text = item['title']+"\n\n"+item['text']
                    doc = {
                        'text': text,
                        'source': item['source'],
                        'language_type': lang,
                        'date': item['date']
                    }
                    fOut.write(json.dumps(doc, ensure_ascii=False) + '\n')
    for root_dir, dirs, files in os.walk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/ta_murasu"):
        for file in files:
            with open(os.path.join(root_dir, file), 'r', encoding='utf8') as f:
                for line in f:
                    item = json.loads(line)
                    text = item['title']+"\n\n"+item['text']
                    doc = {
                        'text': text,
                        'source': item['source'],
                        'language_type': lang,
                        'date': item['date']
                    }
                    fOut.write(json.dumps(doc, ensure_ascii=False) + '\n')


ds=Dataset.from_json("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/ta.jsonl")

ds.save_to_disk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/ta.hf", num_proc=5)
