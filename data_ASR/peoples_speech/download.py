from datasets import load_dataset

ds = load_dataset("MLCommons/peoples_speech", "clean", trust_remote_code=True, cache_dir="/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/.cache", num_proc=16)

ds.save_to_disk("/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/data_ASR/peoples_speech/peoples_speech", num_proc=4)
