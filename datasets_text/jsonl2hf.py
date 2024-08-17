from datasets import load_dataset
from glob import glob


ds=load_dataset("json", data_files=glob("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/non_local/zh/WuDaoCorpus2.0_200G/part-*.jsonl", recursive=True), num_proc=4)
ds.save_to_disk("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/non_local/zh/WuDaoCorpus2.0_200G.hf", num_proc=4)
