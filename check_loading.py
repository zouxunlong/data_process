from datasets import load_from_disk
import os


def get_all_split(root_hf):
    directories = []
    for dirpath, dirnames, filenames in os.walk(root_hf):
        if len(dirnames) == 0:
            directories.append(dirpath)
    return directories

def do_check(ds_path):
    try:
        ds = load_from_disk(ds_path)
    except:
        print(f"error loading {ds_path}", flush=True)



def main(dir):
    splits=get_all_split(dir)
    splits.sort()
    for split in splits:
        do_check(split)


if __name__ == "__main__":
    import fire
    fire.Fire(main)