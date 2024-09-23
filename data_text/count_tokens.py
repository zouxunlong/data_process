import os
from datasets import load_from_disk

print(os.getpid(), flush=True)

ds=load_from_disk("sg_crwal.hf")
print(ds[0])
print(len(ds["tokens"]))
print(sum(ds["tokens"]))

print("all complete", flush=True)

