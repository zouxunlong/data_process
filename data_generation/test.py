from datasets import load_from_disk


ds=load_from_disk("/mnt/data/all_datasets/datasets_multimodal/test/SQA/dream_SQA_v1")

for i, sample in enumerate(ds):
    if sample["instruction"]["text"] == 'No Question Found' or sample['answer']['text'] == 'No Answer Found':
        print(sample, flush=True)
        break
    print(i,flush=True)