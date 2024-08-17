from datasets import load_from_disk
import os

print(os.getpid(), flush=True)
ds = load_from_disk("/home/user/data/data_text/data_skypile-150/SkyPile-150B.hf/train")
print(len(ds), flush=True)
print(ds[0], flush=True)

print("complete", flush=True)
