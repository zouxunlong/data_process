import numpy as np
import json
from fire import Fire
import logging
import soundfile as sf
from multiprocessing import Pool

from tqdm import tqdm



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def mix_wav(param):
    key, file1, file2, start1, start2=param
    
    part=key.split("_")[0]

    array1, sr1 = sf.read(file1.replace("Scripts", "Audio").replace(".TextGrid", ".wav"))
    array2, sr2 = sf.read(file2.replace("Scripts", "Audio").replace(".TextGrid", ".wav"))
    
    length1=len(array1)
    length2=len(array2)

    length=max(length1 + start2, length2 + start1)

    mixed_array = np.concatenate([np.array([0]*start2), array1, np.array([0]*(length-length1-start2))]) + np.concatenate([np.array([0]*start1), array2, np.array([0]*(length-length2-start1))])

    output_file=f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/Audio_mixed/{key}.wav"

    sf.write(output_file, mixed_array, 16000)




def get_key(path):
    part, dir, file=path.split("/")[-3:]
    if part=="PART3":
        return file.split("_")[1]
    if part=="PART4":
        return file.split("_")[1]
    if part=="PART5":
        return file.split("_")[1] + "_" + file.split("_")[4].split(".")[0]
    if part=="PART6":
        return file.split("_")[1] + "_" + file.split("-")[-1].split(".")[0]
    


def mix():

    items=json.load(open("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_shift_selected.json", "r", encoding="utf-8"))

    params=[]

    for key, value in items.items():

        file1, file2 = value.keys()
        start1=value[file1]
        start2=value[file2]
        params.append((key, file1, file2, start1, start2))

    with Pool(processes=56) as pool:
        results = list(tqdm(pool.imap_unordered(mix_wav, params), total=len(params)))
    print(len(params), len(results), flush=True)


def main():
    logging.info(f"Start")
    mix()
    logging.info(f"Done")


if __name__=="__main__":
    logging.info(f"Start")
    Fire(main)
    logging.info(f"Done")
