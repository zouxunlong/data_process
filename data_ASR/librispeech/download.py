from datasets import load_dataset

ds = load_dataset("openslr/librispeech_asr", cache_dir="/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/.cache", num_proc=16)

ds.save_to_disk("/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/data_ASR/librispeech/librispeech_asr", num_proc=4)

