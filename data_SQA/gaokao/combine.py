from datasets import load_from_disk, concatenate_datasets

ds0=load_from_disk("/home/all_datasets/datasets_multimodal/SQA/cn_college_english_exam.schemed")
ds1=load_from_disk("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/gaokao/gaokao.hf")

ds=concatenate_datasets([ds0, ds1])
ds.save_to_disk("gaokao.schemed", num_proc=10)