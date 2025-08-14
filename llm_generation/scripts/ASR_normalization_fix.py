
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


def fix(split, num_proc=112):

    print("start {}".format(split), flush=True)

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

    if os.path.exists(split.replace("/datasets_hf_stage_AudioLLM_v2_normalized/", "/datasets_hf_stage_AudioLLM_v2_normalized_fixed/")):
        print("completed {}".format(split), flush=True)
        return

    ds.save_to_disk(split.replace("/datasets_hf_stage_AudioLLM_v2_normalized/", "/datasets_hf_stage_AudioLLM_v2_normalized_fixed/"))
    print("complete {}".format(split), flush=True)


def main(pattern="/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2_normalized/datasets_multimodal/train/ASR/*"):
    splits = glob(pattern)
    splits.sort(reverse=False)
    pprint(splits)
    for split in splits:
        if os.path.exists(split.replace("/datasets_hf_stage_AudioLLM_v2_normalized/", "/datasets_hf_stage_AudioLLM_v2_normalized_fixed/")):
            print("completed {}".format(split), flush=True)
            continue
        fix(split)


if __name__ == '__main__':
    fire.Fire(main)
