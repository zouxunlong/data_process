from datasets import load_from_disk


def filter_0(dataset_path):
    def filter_fn(batch):
        return [False if sample.strip() else True for sample in  batch["input"]]

    ds = load_from_disk(dataset_path)
    print("original samples: {}".format(len(ds)), flush=True)
    ds=ds.filter(function=filter_fn, batched=True, num_proc=10)
    print("filtered samples: {}".format(len(ds)), flush=True)
    ds.save_to_disk(dataset_path+".filtered0", num_proc=4)


def filter_1(dataset_path):
    def filter_fn(batch):
        return [False if "=" in sample else True for sample in  batch["instruction"]]

    ds = load_from_disk(dataset_path)
    print("original samples: {}".format(len(ds)), flush=True)
    ds=ds.filter(function=filter_fn, batched=True, num_proc=10)
    print("filtered samples: {}".format(len(ds)), flush=True)
    ds.save_to_disk(dataset_path+".filtered1", num_proc=4)


if __name__ == "__main__":
    dataset_path = "instruction/openhermes.filtered0.filtered1"
    filter_1(dataset_path)
