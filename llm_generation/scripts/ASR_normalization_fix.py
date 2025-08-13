
from pprint import pprint
import fire
from datasets import load_from_disk
from glob import glob
import os


def map_fn(batch):

    original_answers = [other_attributes['original_answer'] for other_attributes in batch['other_attributes']]
    need_futher_process = [other_attributes['need_further_process'] for other_attributes in batch['other_attributes']]

    for i, (original_answer, need_process) in enumerate(zip(original_answers, need_futher_process)):
        if need_process:
            batch['answer'][i]['text'] = original_answer

    return batch


def fix(split, num_proc=32):

    ds = load_from_disk(split)

    features = ds.features

    if 'need_further_process' not in features["other_attributes"]:
        print("no need to fix {}".format(split), flush=True)
        return

    ds = ds.map(
        map_fn,
        features          = features,
        batched           = True,
        batch_size        = 100,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "fix for {}".format(split),
    )

    ds.save_to_disk(split.replace("datasets_hf_stage_AudioLLM_v2", "datasets_hf_stage_AudioLLM_v2_fixed"), num_proc=4)


def main(pattern="/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/*"):
    splits = glob(pattern)
    splits.sort(reverse=True)
    pprint(splits)
    for split in splits:
        if os.path.exists(split.replace("datasets_hf_stage_AudioLLM_v2", "datasets_hf_stage_AudioLLM_v2_fixed")):
            print("completed {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        fix(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
