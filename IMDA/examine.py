from datasets import load_from_disk
import soundfile as sf
from fire import Fire
import json

def examine(*args):
    for dataset_path in args:
        ds=load_from_disk(dataset_path)

        for i in [0, 1, 403, 1403, 1903]:
            item=ds[i]
            print(i, flush=True)
            sf.write("{}.wav".format(i), item["context"]["audio"]["array"], 16000)
            with open("{}.txt".format(i), "w", encoding="utf8") as f_out:
                del item["context"]
                f_out.write(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    Fire(examine)
