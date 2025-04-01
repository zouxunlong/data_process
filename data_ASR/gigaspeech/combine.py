from datasets import load_from_disk, concatenate_datasets

ds_paths = ["/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/chinese_callcenter_datatang_500hrs_30_zh_300_ASR",
            "/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/chinese_callcenter_datatang_500hrs_300_zh_300_ASR"
            ]
print("start concat", flush=True)

ds = concatenate_datasets([load_from_disk(path) for path in ds_paths])
print("start save", flush=True)
ds.save_to_disk(
    "/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/chinese_callcenter_datatang_500hrs_zh_300_ASR", num_proc=4)
