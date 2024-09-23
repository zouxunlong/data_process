from datasets import Dataset

ds=Dataset.from_json("zh.local.jsonl")
ds.save_to_disk("zh.local.hf", num_proc=5)
