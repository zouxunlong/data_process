from datasets import load_from_disk
import os


def get_all_split(root_hf):
    directories = []
    for dirpath, dirnames, filenames in os.walk(root_hf):
        if len(dirnames) == 0:
            directories.append(dirpath)
    return directories



def main(dir):
    splits=get_all_split(dir)
    splits.sort()
    for split in splits:
        print(split, flush=True)


if __name__ == "__main__":
    import fire
    fire.Fire(main)