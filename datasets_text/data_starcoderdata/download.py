from datasets import load_dataset
import os

print(os.getpid(), flush=True)
dataset = load_dataset(path="bigcode/starcoderdata", data_dir="python", split="train")
for split, ds in dataset.items():
    ds.save_to_disk("starcoderdata.hf/{}".format(split), num_proc=10)

print("complete", flush=True)

