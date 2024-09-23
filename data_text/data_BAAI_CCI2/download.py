from datasets import load_dataset
import os

print(os.getpid(), flush=True)
dataset = load_dataset("BAAI/CCI2-Data")
for split, ds in dataset.items():
    ds.save_to_disk("cci2.hf/{}".format(split), num_proc=10)

print("complete", flush=True)