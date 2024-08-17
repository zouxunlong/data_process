from datasets import Dataset

ds=Dataset.from_json("/home/user/data/newslink_data/newslink_zh.jsonl")
ds.save_to_disk("/home/user/data/data_text/data_newslink/newslink_zh.hf", num_proc=5)
