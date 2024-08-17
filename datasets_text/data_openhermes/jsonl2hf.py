from datasets import Dataset

ds=Dataset.from_json("/home/user/data/data_text/data_openhermes/OpenHermes-2.5/openhermes2_5.json")
ds.save_to_disk("/home/user/data/data_text/data_openhermes/OpenHermes-2.5/openhermes2_5.hf", num_proc=5)
