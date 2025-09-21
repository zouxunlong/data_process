from glob import glob
from datasets import load_from_disk
import random

instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


def map_fn(batch):

    instructions = [random.choice(instructions_asr) for _ in batch["instruction"]]
    batch["instruction"]=instructions
    return batch


def convert(split):
    ds = load_from_disk(split)
    ds = ds.map(map_fn, batched=True, num_proc=14)
    ds.save_to_disk(split.replace("/datasets_hf_stage_AudioLLM_v3/", "/datasets_hf_stage_AudioLLM_v3_updated/"), num_proc=1)


def main():
    splits=glob("/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v3/datasets_multimodal/*/ST/*_hok_zh_30_ST")
    print(splits, flush=True)
    for split in splits:
        print(f"start {split}")
        convert(split)


if __name__ == "__main__":
    main()

