from datasets import load_dataset

ds = load_dataset("Hemanth-thunder/tamil-madlad-400")

ds.save_to_disk("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/data_text/data_tamil-madlad-400/tamil-madlad-400", num_proc=4)

print("Done!")