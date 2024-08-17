import os
from datasets import load_from_disk

for dir in [ f.path for f in os.scandir("./CoVoST2") if f.is_dir() ]:
    audio_dataset= load_from_disk(dir)
    for split, ds in audio_dataset.items():
        print(split, flush=True)
        for item in ds:
            print(item, flush=True)
            break
