from glob import glob
from datasets import load_dataset


data_files=glob("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/data_text/data_fineweb-edu/fineweb-edu/data/CC-MAIN-202*/*.parquet", recursive=True)
data_files.sort()


ds = load_dataset("parquet", data_files=data_files, num_proc=4)

ds.save_to_disk("./fineweb-edu.hf/2020_2024", num_proc=8)
