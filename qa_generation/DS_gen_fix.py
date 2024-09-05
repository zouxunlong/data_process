
from glob import glob
import os
import fire
from datasets import load_from_disk


def ds_generation(split):

    ds = load_from_disk(split)

    ds = ds.filter(lambda x: x['answer']['text'] != 'Template not matched.',
                   batch_size=1,
                   writer_batch_size=1,
                   num_proc=20
                   )

    ds.save_to_disk(split.replace("_DS_v1", "_DS_v2"), num_proc=4)


def main(pattern):
    splits = glob(pattern)
    splits.sort()
    print(splits, flush=True)
    for split in splits:
        if os.path.exists(split.replace("_DS_v1", "_DS_v2")):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        ds_generation(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
