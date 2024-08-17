from collections import defaultdict
import json
import os
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets


def get_length(dataset_path):
    length_distribution=defaultdict(int)
    ds=load_from_disk(dataset_path)

    for i, length in enumerate(ds["length"]):
        length_distribution[(length//100)*100] += 1
        if i%50000==0:
            print(i, flush=True)
    dfList = []
    for key, value in length_distribution.items():
        dfList.append([key, value])
    df_new =  pd.DataFrame(dfList)
    df_new.to_excel(dataset_path+'.xlsx', index=False, header=False)


if __name__ == "__main__":
    print(os.getpid(), flush=True)
    get_length()
    print("finished all", flush=True)
