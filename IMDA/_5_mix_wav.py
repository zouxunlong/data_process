from collections import defaultdict
from glob import glob
from itertools import groupby
import os
import json
import numpy as np
from fire import Fire
import logging
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_key(path):
    elements = path.split("/")[-1].replace("-", "_").split("_")
    return elements[1]+"_"+elements[-1]


def mix_wav(params):
    input_files, output_file, same_time = params

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    assert sr1 == sr2 == 16000, f"{input_files[0]}:{sr1}, {input_files[1]}:{sr2}"

    if same_time=="same_time":
        length = min(len(array1), len(array2))
        sf.write(output_file, array1[0:length] + array2[0:length], 16000)
        return (input_files[0], input_files[1], 0, 0)
    if same_time=="diff_time":
        length_diff = len(array1) - len(array2)
        max_agv_sum = 0

        for shift in range(min(-160000, -160000+length_diff), max(160000, length_diff + 160000), 320):
            mixed_array        = array1[max(0, shift):min(len(array1), len(array2) + shift)] + array2[max(0, -shift):min(len(array1)-shift, len(array2))]
            mixed_array_length = len(mixed_array)

            shifted_abs_sum = np.sum(np.absolute(mixed_array))
            shifted_avg_sum = shifted_abs_sum / mixed_array_length

            if shifted_avg_sum > max_agv_sum:
                max_agv_sum         = shifted_avg_sum
                optimal_shift       = shift
                optimal_mixed_array = mixed_array
        sf.write(output_file, optimal_mixed_array, 16000)
        return (input_files[0], input_files[1], max(0, optimal_shift), max(0, -optimal_shift))


def mix_part3():

    def get_key(path):
        return path.split("/")[-1].split("_")[1]

    root_out      = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/Audio_Separate_mixed"
    duration_dict = json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/duration_dict.json", "r", encoding="utf-8"))

    root      = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/Audio_Separate"
    wav_files = glob(os.path.join(root, '*.wav'), recursive=True)
    wav_files.sort()


    params = []
    for key, value in groupby(wav_files, key=get_key):

        try:
            audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
        except ValueError as e:
            logging.error(f"Error: {key}")
            continue

        wav_time1    = duration_dict["Audio_Separate"][audio_file1]
        wav_time2    = duration_dict["Audio_Separate"][audio_file2]

        if wav_time1 == wav_time2:
            params.append(([os.path.join(root, audio_file1), os.path.join(root, audio_file2)], os.path.join(root_out, f"{key}.wav"), "same_time"))
        else:
            params.append(([os.path.join(root, audio_file1), os.path.join(root, audio_file2)], os.path.join(root_out, f"{key}.wav"), "diff_time"))

    shift_dict = defaultdict(float)
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap_unordered(mix_wav, params), total=len(params)))
        for file1, file2, shift1, shift2 in results:
            shift_dict[file1.replace(".wav", ".TextGrid")].update(shift1)
            shift_dict[file2.replace(".wav", ".TextGrid")].update(shift2)
        json.dump(shift_dict, open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/mix_shift.json", "w"), indent=4)


def mix_part4():

    def get_key(path):
        return path.split("/")[-1].split("_")[1]

    root_out="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/Audio_mixed"
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/duration_dict.json", "r", encoding="utf-8"))

    root="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/Audio"
    wav_files = glob(os.path.join(root, '*.wav'), recursive=True)
    wav_files.sort()


    with Pool(processes=32) as pool:
        params = []
        keys=set()
        
        for key, value in groupby(wav_files, key=get_key):
            if key in keys or os.path.exists(os.path.join(root_out, f"{key}.wav")):
                continue

            try:
                audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
            except ValueError as e:
                logging.error(f"Error: {key}")
                continue


            wav_time1    = duration_dict["Audio"][audio_file1]
            wav_time2    = duration_dict["Audio"][audio_file2]
            script_time1 = duration_dict["Scripts"][audio_file1.split(".")[0]+".TextGrid"]
            script_time2 = duration_dict["Scripts"][audio_file2.split(".")[0]+".TextGrid"]

            if abs(wav_time1-wav_time2)<0.1:
                params.append(([os.path.join(root, audio_file1), os.path.join(root, audio_file2)], os.path.join(root_out, f"{key}.wav")))
                keys.add(key)
            else:
                logging.error(f"unmatch length: {key} || audio_file1:{int(wav_time1)}:{int(script_time1)}, audio_file2: {int(wav_time2)}:{int(script_time2)}")

        results = list(tqdm(pool.imap_unordered(mix_wav, params), total=len(params)))
        print(len(keys), len(params), len(results), flush=True)



def mix_part5():

    def get_key(path):
        return path.split("/")[-1].split("_")[1] + "_" + path.split("/")[-1].split("_")[4].split(".")[0]

    root_out="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5/Audio_mixed"
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5/duration_dict.json", "r", encoding="utf-8"))

    root="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5/Audio"
    wav_files = glob(os.path.join(root, '*.wav'), recursive=True)
    wav_files.sort(key=get_key)

    with Pool(processes=32) as pool:
        params = []
        keys=set()
        
        for key, value in groupby(wav_files, key=get_key):
            if key in keys or os.path.exists(os.path.join(root_out, f"{key}.wav")):
                continue

            try:
                audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
            except ValueError as e:
                logging.error(f"Error: {key}")
                continue

            wav_time1    = duration_dict["Audio"][audio_file1]
            wav_time2    = duration_dict["Audio"][audio_file2]
            script_time1 = duration_dict["Scripts"][audio_file1.split(".")[0]+".TextGrid"]
            script_time2 = duration_dict["Scripts"][audio_file2.split(".")[0]+".TextGrid"]

            if abs(wav_time1-wav_time2)<0.1:
                params.append(([os.path.join(root, audio_file1), os.path.join(root, audio_file2)], os.path.join(root_out, f"{key}.wav")))
                keys.add(key)
            else:
                logging.error(f"unmatch length: {key} || audio_file1:{int(wav_time1)}:{int(script_time1)}, audio_file2: {int(wav_time2)}:{int(script_time2)}")

        results = list(tqdm(pool.imap_unordered(mix_wav, params), total=len(params)))
        print(len(keys), len(params), len(results), flush=True)



def mix_part6():

    def get_key(path):
        return path.split("/")[-1].split("_")[1] + "_" + path.split("/")[-1].split("-")[-1].split(".")[0]

    root_out="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/Audio_mixed"
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/duration_dict.json", "r", encoding="utf-8"))

    root="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/Audio"
    wav_files = glob(os.path.join(root, '*.wav'), recursive=True)
    wav_files.sort(key=get_key)
    
    with Pool(processes=32) as pool:
        params = []
        keys=set()
        
        for key, value in groupby(wav_files, key=get_key):
            if key in keys or os.path.exists(os.path.join(root_out, f"{key}.wav")):
                continue

            try:
                audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
            except ValueError as e:
                logging.error(f"Error: {key}")
                continue

            wav_time1    = duration_dict["Audio"][audio_file1]
            wav_time2    = duration_dict["Audio"][audio_file2]
            script_time1 = duration_dict["Scripts"][audio_file1.split(".")[0]+".TextGrid"]
            script_time2 = duration_dict["Scripts"][audio_file2.split(".")[0]+".TextGrid"]

            if abs(wav_time1-wav_time2)<0.1:
                params.append(([os.path.join(root, audio_file1), os.path.join(root, audio_file2)], os.path.join(root_out, f"{key}.wav")))
                keys.add(key)
            else:
                logging.error(f"unmatch length: {key} || audio_file1:{int(wav_time1)}:{int(script_time1)}, audio_file2: {int(wav_time2)}:{int(script_time2)}")

        results = list(tqdm(pool.imap_unordered(mix_wav, params), total=len(params)))
        print(len(keys), len(params), len(results), flush=True)


def main():
    logging.info(f"Start part3")
    mix_part3()
    # logging.info(f"Start part4")
    # mix_part4()
    # logging.info(f"Start part5")
    # mix_part5()
    # logging.info(f"Start part6")
    # mix_part6()


if __name__=="__main__":
    logging.info(f"Start")
    Fire(main)
    logging.info(f"Done")
