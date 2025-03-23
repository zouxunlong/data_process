from datasets import load_from_disk, concatenate_datasets

ds_paths = [f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/datasets_hf_with_length/train/ASR/wenetspeech_with_length/wenetspeec_{i}" for i in range(30)]
print("start concat", flush=True)

ds = concatenate_datasets([load_from_disk(path) for path in ds_paths])
print("start save", flush=True)
ds.save_to_disk(
    "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/datasets_hf_with_length/train/ASR/wenetspeech_with_length2", num_proc=20)
