from datasets import concatenate_datasets, load_dataset, load_from_disk
import os


def get_all_split(ds_path):
    directories = []
    for dirpath, dirs, files in os.walk(ds_path):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort()
    return directories


def select_sg_no_covid(
    ds_path="/mnt/home/zoux/datasets/fineweb/fineweb/data",
    num_proc=140):

    def filter_fn(batch):
        return [True if "singapore" in text.lower() and not "covid" in text.lower() else False for text in batch["text"]]

    splits=get_all_split(ds_path)
    for i, split in enumerate(splits):
        if os.path.exists(split.replace("/data/", "/fineweb.sg.no_covid/")):
            continue
        print("starting {}".format(os.path.basename(split)), flush=True)
        ds = load_dataset("parquet", data_files=split+"/*.parquet", num_proc=num_proc)["train"].select_columns(["text", "date", "token_count"]).rename_column("token_count", "tokens")
        print(ds.column_names, flush=True)
        ds = ds.filter(filter_fn, batched=True, num_proc=num_proc, desc="filtering {}th segment {}".format(i, os.path.basename(split)))
        ds.save_to_disk(split.replace("/data/", "/fineweb.sg.no_covid/"), num_proc=6)
        print(ds.column_names, flush=True)
        print("complete {}".format(os.path.basename(split)), flush=True)

    print("complete all", flush=True)


def combine(dir):
    splits=get_all_split(dir)
    ds=concatenate_datasets([load_from_disk(split) for split in splits])
    ds.save_to_disk(dir.replace("fineweb-edu.sg", "fineweb-edu.sg.combined"), num_proc=4)


if __name__ == "__main__":
    select_sg_no_covid()

