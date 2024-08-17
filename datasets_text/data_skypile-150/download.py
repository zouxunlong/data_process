from datasets import load_dataset
import os

print(os.getpid(), flush=True)
dataset = load_dataset("Skywork/SkyPile-150B")
for split, ds in dataset.items():
    ds.save_to_disk("SkyPile-150B.hf/{}".format(split), num_proc=10)

print("complete", flush=True)