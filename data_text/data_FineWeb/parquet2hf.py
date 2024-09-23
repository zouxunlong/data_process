from glob import glob
from datasets import load_dataset


ds = load_dataset("parquet", data_files=glob("/home/user/data/data_text/data_FineWeb/fineweb/data/CC-MAIN-*/*.parquet", recursive=True), num_proc=24)

ds.save_to_disk("/home/user/data/data_text/data_FineWeb/fineweb.hf", num_proc=24)
