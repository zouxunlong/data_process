from datasets import load_from_disk, concatenate_datasets


ds = concatenate_datasets([load_from_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes/train/DS/IMDA_PART6_30_DS_v4_{i}") for i in range(10)])
ds.save_to_disk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes/train/DS/IMDA_PART6_30_DS_v4", num_proc=4)
print(f"Done")

