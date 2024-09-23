from datasets import load_dataset
from glob import glob


ds=load_dataset("json", data_files=glob("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/ta_*/*.jsonl", recursive=True), num_proc=4)
ds.save_to_disk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/newslink_ta.hf", num_proc=4)
