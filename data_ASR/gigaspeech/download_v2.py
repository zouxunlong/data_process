from datasets import load_dataset

ds = load_dataset("speechcolab/gigaspeech2",
                  cache_dir="/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/.cache", 
                  num_proc=16)

ds.save_to_disk(
    "/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/data_ASR/gigaspeech/gigaspeech2",
    num_proc=4)
