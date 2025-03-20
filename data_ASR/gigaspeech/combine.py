from datasets import load_from_disk, concatenate_datasets

ds_paths = [
    # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_0_2",
    # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_2_5",
    # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_5_10",
    # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_10_13",
    # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_13_16",
    # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_16_20",
    "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_0_10",
    "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train_10_20"
]

ds = concatenate_datasets([load_from_disk(path) for path in ds_paths])
ds.save_to_disk(
    "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/gigaspeech2/id_new_bytes/train", num_proc=20)
