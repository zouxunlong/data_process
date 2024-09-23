from datasets import load_dataset
import os

print(os.getpid(), flush=True)
ds = load_dataset("HuggingFaceFW/fineweb-edu", num_proc=10)

for split, ds in ds.items():
    ds.save_to_disk("fineweb-edu-hf/{}".format(split), num_proc=10)

print("complete", flush=True)