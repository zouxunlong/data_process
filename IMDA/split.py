from datasets import load_from_disk
import random
from fire import Fire

def split(ds_path, select_num):

    ds = load_from_disk(ds_path)
    all = len(ds)

    selected = set(random.sample(range(all), select_num))
    lefted = set(range(all)) - selected

    print(len(selected), flush=True)
    print(len(lefted), flush=True)
    print("start split", flush=True)

    test_set = ds.select(indices=list(selected))
    test_set.save_to_disk(ds_path+".test", num_proc=6)
    print("complet saving test", flush=True)

    train_set = ds.select(indices=list(lefted))
    train_set.save_to_disk(ds_path+".train", num_proc=6)
    print("complet saving train", flush=True)


if __name__=="__main__":
    Fire(split)
