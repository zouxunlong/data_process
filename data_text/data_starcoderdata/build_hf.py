from datasets import load_dataset, concatenate_datasets
import os
from glob import glob

print(os.getpid(), flush=True)
root="/home/user/data/data_text/data_starcoderdata/starcoderdata"
dirs=sorted(glob(root+"/*/", recursive = True))
for dir in dirs:
    parquet_paths=glob(os.path.join(dir, "*.parquet"), recursive=True)
    ds = concatenate_datasets([load_dataset("parquet", data_files=parquet_path)["train"] for parquet_path in parquet_paths])
    print(len(ds), flush=True)
    print(ds.column_names, flush=True)
    ds.save_to_disk(dir.replace("/starcoderdata", "/starcoderdata.hf"), num_proc=4)

print("complete", flush=True)
