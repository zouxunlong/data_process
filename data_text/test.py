from datasets import load_dataset, load_from_disk

ds = load_from_disk("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/datasets_text/data_text/data_fineweb-edu/fineweb-edu.sg/2")

for i, item in enumerate(ds):
    print(item["text"])
    print("========================================")
    if i==10:
        break