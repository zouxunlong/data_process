from datasets import load_from_disk
import os

print(os.getpid(), flush=True)
ds = load_from_disk("CCI2-Data.hf/cci2")
for item in ds:
    print(item, flush=True)
    break

print("complete", flush=True)
