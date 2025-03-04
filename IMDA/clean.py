from datasets import load_from_disk
from tqdm import tqdm


def clean(test_path, workers=56):

    train_path = test_path.replace("/test/", "/train/")
    ds_test    = load_from_disk(test_path)
    ds_train   = load_from_disk(train_path)
    print(ds_test, flush=True)
    print(ds_train, flush=True)

    ids = []
    for sample in tqdm(ds_test["other_attributes"]):
        ids.append(sample["id"])
    print(len(ids), flush=True)
    print(len(set(ids)), flush=True)

    print(len(ds_train), flush=True)
    ds_train = ds_train.filter(lambda x: [other_attribute["id"] not in set(ids) for other_attribute in x["other_attributes"]],
                         batched=True, batch_size=1000, writer_batch_size=1000, num_proc=workers)
    print(len(ds_train), flush=True)

    ds_train.save_to_disk(train_path+"_v5", num_proc=10)


def main():
    splits = [
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART1_AR",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART1_GR",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART1_MIX",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART2_AR",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART2_GR",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART2_MIX",
    ]
    for test_path in splits:
        print("start", test_path, flush=True)
        clean(test_path)
    print("complete all", flush=True)


if __name__ == "__main__":
    main()
