import os
from time import time
from datasets import load_from_disk
from fire import Fire
from multiprocessing import Pool
from tqdm import tqdm

def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def check_entry(args):
    ds, start, end, worker = args
    problem_ids=[]
    for idx in tqdm(range(start, end), desc=f"Processing {start}-{end}", position=worker):
        try:
            sample=ds[idx]
        except:
            problem_ids.append(idx)
    return problem_ids


def main():

    splits = get_all_split("/mnt/data/all_datasets/datasets_multimodal/train/ST/CoVoST2_v1_ta_en")
    splits.sort()

    start=time()
    for split in splits:
        print("start {}".format(split), flush=True)
        ds = load_from_disk(split)
        N=len(ds)
        print(f"total {N}", flush=True)
        problem_ids=[]
        for i in tqdm(range(len(ds))):
            try:
                sample=ds[i]
            except:
                problem_ids.append(i)
        print(f"errors: {len(problem_ids)}", flush=True)
        ds = ds.select([i for i in range(len(ds)) if i not in problem_ids])
        print(f"rest {len(ds)}", flush=True)
    print(f"duration {time()-start}", flush=True)


    # start=time()
    # for split in splits:
    #     print("start {}".format(split), flush=True)
    #     ds = load_from_disk(split)
    #     N=len(ds)
    #     chunk_size = (N + 120 - 1) // 120
    #     print(f"total {N}", flush=True)
    #     pool_inputs = [(ds, i, min(i + chunk_size, N), worker) for worker, i in enumerate(range(0, len(ds), chunk_size))]
    #     with Pool(processes=120) as pool:
    #         results = pool.map(check_entry, pool_inputs)
    #     problem_ids = [item for sublist in results for item in sublist]
    #     print(f"errors: {len(problem_ids)}", flush=True)
    #     ds = ds.select([i for i in range(len(ds)) if i not in problem_ids])
    #     print(f"rest {len(ds)}", flush=True)
    # print(f"duration {time()-start}", flush=True)

if __name__ == '__main__':
    Fire(main)

