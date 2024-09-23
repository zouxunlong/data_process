from datasets import Dataset, Features, Value, load_dataset

ds=load_dataset("json", data_files="/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.en.jsonl")
ds.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc_en", num_proc=5)

ds=load_dataset("json", data_files="/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.id.jsonl")
ds.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc_id", num_proc=5)

ds=load_dataset("json", data_files="/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.zh.jsonl")
ds.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc_zh", num_proc=5)
