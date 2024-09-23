from datasets import load_from_disk
from fire import Fire


def deduplicate(dataset_path, output_path):

    def filter_fn(example, uniques):
        if example["simhash"] in uniques:
            uniques.remove(example["simhash"])
            return True
        else:
            return False

    ds = load_from_disk(dataset_path)
    uniques = set(ds.unique("simhash"))
    ds = ds.filter(filter_fn, fn_kwargs={"uniques": uniques}, load_from_cache_file=False) # num_proc=1, or else, will result in incomplete deduplication
    ds.save_to_disk(output_path, num_proc=4)


if __name__ == "__main__":

    Fire(deduplicate)

