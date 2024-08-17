# from datasets import load_dataset

# dataset = load_dataset("d0rj/audiocaps")

# for split, dataset in dataset.items():
#     print(split,flush=True)
#     print(dataset[0])

from datasets import load_dataset

dataset = load_dataset("agkphysics/AudioSet")
dataset.save_to_disk("AudioSet_hf")
for split, split_dataset in dataset.items():
    print(split,flush=True)
    print(len(split_dataset), flush=True)
    split_dataset.to_json("./AudioSet_jsonl/AudioSet_{}.jsonl".format(split))
