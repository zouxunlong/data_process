from datasets import load_from_disk

ds = load_from_disk("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id/train")

# Number of subsets
num_subsets = 20

# Split the dataset into 20 subsets
subsets = [ds.shard(num_shards=num_subsets, index=i) for i in range(num_subsets)]

for i, subset in enumerate(subsets):
    subset.save_to_disk(f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id/train_{i}", num_proc=8)
