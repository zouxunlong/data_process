from datasets import load_dataset

ds = load_dataset("asapp/slue-phase-2", "sqa5",
                  cache_dir="/mnt/data/all_datasets/xunlong_working_repo/.cache",
                  num_proc=16)
ds.save_to_disk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/slue-phase-2/SLUE_Phase_2_SQA",
                num_proc=4)
