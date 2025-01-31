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



def mix_wav_large(params_large):
    input_files, output_file, same_time, part = params_large
    print(output_file, flush=True)
    f_out=f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/mix_shift.jsonl"

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    assert sr1 == sr2 == 16000, f"{input_files[0]}:{sr1}, {input_files[1]}:{sr2}"

    if os.path.basename(output_file) in ["2685.wav", "2727.wav", "1103.wav"]:
        length = min(len(array1), len(array2))
        sf.write(output_file, array1[0:length] + array2[0:length], 16000)
        open(f_out, "a").write(json.dumps({input_files[0]: 0}) + "\n")
        open(f_out, "a").write(json.dumps({input_files[1]: 0}) + "\n")
        return (input_files[0], input_files[1], 0, 0)

    if os.path.basename(output_file) in ["0809.wav"]:
        start1 = max(0, -8000)
        end1   = min(len(array1), len(array2) - 8000)
        start2 = max(0, 8000)
        end2   = min(len(array1) + 8000, len(array2))
        sf.write(output_file, array1[start1:end1] + array2[start2:end2], 16000)
        open(f_out, "a").write(json.dumps({input_files[0]: 0}) + "\n")
        open(f_out, "a").write(json.dumps({input_files[1]: 8000}) + "\n")
        return (input_files[0], input_files[1], 0, 8000)

    if same_time=="diff_time":
        length_diff = len(array1) - len(array2)
        max_agv_sum = 0

        shift_range = range(min(-80000, -80000 + length_diff), max(80000, length_diff + 80000), 1600)

        for shift in shift_range:
            start1 = max(0, shift)
            end1   = min(len(array1), len(array2) + shift)
            start2 = max(0, -shift)
            end2   = min(len(array1) - shift, len(array2))

            mixed_array = array1[start1:end1] + array2[start2:end2]
            shifted_avg_sum = np.mean(np.abs(mixed_array))

            if shifted_avg_sum > max_agv_sum:
                max_agv_sum         = shifted_avg_sum
                optimal_shift       = shift
                optimal_mixed_array = mixed_array

        sf.write(output_file, optimal_mixed_array, 16000)
        shift1 = max(0, optimal_shift)
        shift2 = max(0, -optimal_shift)
        open(f_out, "a").write(json.dumps({input_files[0]: shift1}) + "\n")
        open(f_out, "a").write(json.dumps({input_files[1]: shift2}) + "\n")
        return (input_files[0], input_files[1], shift1, shift2)





def mix_wav(params):
    input_files, output_file, same_time, part = params
    f_out=f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/mix_shift.jsonl"

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    assert sr1 == sr2 == 16000, f"{input_files[0]}:{sr1}, {input_files[1]}:{sr2}"

    if same_time=="same_time":
        length = min(len(array1), len(array2))
        sf.write(output_file, array1[0:length] + array2[0:length], 16000)
        open(f_out, "a").write(json.dumps({input_files[0]: 0}) + "\n")
        open(f_out, "a").write(json.dumps({input_files[1]: 0}) + "\n")
        return (input_files[0], input_files[1], 0, 0)
    if same_time=="diff_time":
        length_diff = len(array1) - len(array2)
        max_agv_sum = 0

        shift_range = range(min(-80000, -80000 + length_diff), max(80000, length_diff + 80000), 800)

        for shift in shift_range:
            start1 = max(0, shift)
            end1   = min(len(array1), len(array2) + shift)
            start2 = max(0, -shift)
            end2   = min(len(array1) - shift, len(array2))

            mixed_array = array1[start1:end1] + array2[start2:end2]
            shifted_avg_sum = np.mean(np.abs(mixed_array))

            if shifted_avg_sum > max_agv_sum:
                max_agv_sum         = shifted_avg_sum
                optimal_shift       = shift
                optimal_mixed_array = mixed_array

        sf.write(output_file, optimal_mixed_array, 16000)
        shift1 = max(0, optimal_shift)
        shift2 = max(0, -optimal_shift)
        open(f_out, "a").write(json.dumps({input_files[0]: shift1}) + "\n")
        open(f_out, "a").write(json.dumps({input_files[1]: shift2}) + "\n")
        return (input_files[0], input_files[1], shift1, shift2)


def get_key_part3(path):
    return path.split("/")[-1].split("_")[1]

def get_key_part4(path):
    return path.split("/")[-1].split("_")[1]

def get_key_part5(path):
    return path.split("/")[-1].split("_")[1] + "_" + path.split("/")[-1].split("_")[4].split(".")[0]

def get_key_part6(path):
    return path.split("/")[-1].split("_")[1] + "_" + path.split("/")[-1].split("-")[-1].split(".")[0]



def mix():

    params = []
    params_large = []

    root = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw"

    print("Start PART3", flush=True)

    root_mix_part3 = f"{root}/PART3/Audio_Separate_mixed"
    duration_dict  = json.load(open(f"{root}/PART3/duration_dict.json", "r", encoding="utf-8"))

    root_part3 = f"{root}/PART3/Audio_Separate"
    wav_files  = glob(os.path.join(root_part3, '*.wav'), recursive=True)
    wav_files.sort(key=get_key_part3)


    for key, value in groupby(wav_files, key=get_key_part3):
        if os.path.exists(os.path.join(root_mix_part3, f"{key}.wav")):
            continue

        try:
            audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
        except ValueError as e:
            logging.error(f"Error: {key}")
            continue

        wav_time1    = duration_dict["Audio_Separate"][audio_file1]
        wav_time2    = duration_dict["Audio_Separate"][audio_file2]

        if abs(wav_time1 - wav_time2)<0.02:
            params.append(([os.path.join(root_part3, audio_file1), os.path.join(root_part3, audio_file2)], os.path.join(root_mix_part3, f"{key}.wav"), "same_time", "PART3"))
        elif abs(wav_time1 - wav_time2)<200:
            params.append(([os.path.join(root_part3, audio_file1), os.path.join(root_part3, audio_file2)], os.path.join(root_mix_part3, f"{key}.wav"), "diff_time", "PART3"))
        else:
            params_large.append(([os.path.join(root_part3, audio_file1), os.path.join(root_part3, audio_file2)], os.path.join(root_mix_part3, f"{key}.wav"), "diff_time", "PART3"))




    print("Start PART4", flush=True)

    root_mix_part4 = f"{root}/PART4/Audio_mixed"
    duration_dict  = json.load(open(f"{root}/PART4/duration_dict.json", "r", encoding="utf-8"))

    root_part4 = f"{root}/PART4/Audio"
    wav_files  = glob(os.path.join(root_part4, '*.wav'), recursive=True)
    wav_files.sort(key=get_key_part4)

    for key, value in groupby(wav_files, key=get_key_part4):
        if os.path.exists(os.path.join(root_mix_part4, f"{key}.wav")):
            continue
        
        try:
            audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
        except ValueError as e:
            logging.error(f"Error: {key}")
            continue

        wav_time1    = duration_dict["Audio"][audio_file1]
        wav_time2    = duration_dict["Audio"][audio_file2]

        if abs(wav_time1 - wav_time2)<0.02:
            params.append(([os.path.join(root_part4, audio_file1), os.path.join(root_part4, audio_file2)], os.path.join(root_mix_part4, f"{key}.wav"), "same_time", "PART4"))
        elif abs(wav_time1 - wav_time2)<200:
            params.append(([os.path.join(root_part4, audio_file1), os.path.join(root_part4, audio_file2)], os.path.join(root_mix_part4, f"{key}.wav"), "diff_time", "PART4"))
        else:
            params_large.append(([os.path.join(root_part4, audio_file1), os.path.join(root_part4, audio_file2)], os.path.join(root_mix_part4, f"{key}.wav"), "diff_time", "PART4"))




    print("Start PART5", flush=True)

    root_mix_part5 = f"{root}/PART5/Audio_mixed"
    duration_dict  = json.load(open(f"{root}/PART5/duration_dict.json", "r", encoding="utf-8"))

    root_part5 = f"{root}/PART5/Audio"
    wav_files  = glob(os.path.join(root_part5, '*.wav'), recursive=True)
    wav_files.sort(key=get_key_part5)

    for key, value in groupby(wav_files, key=get_key_part5):
        if os.path.exists(os.path.join(root_mix_part5, f"{key}.wav")):
            continue

        try:
            audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
        except ValueError as e:
            logging.error(f"Error: {key}")
            continue

        wav_time1    = duration_dict["Audio"][audio_file1]
        wav_time2    = duration_dict["Audio"][audio_file2]

        if abs(wav_time1 - wav_time2)<0.02:
            params.append(([os.path.join(root_part5, audio_file1), os.path.join(root_part5, audio_file2)], os.path.join(root_mix_part5, f"{key}.wav"), "same_time", "PART5"))
        elif abs(wav_time1 - wav_time2)<200:
            params.append(([os.path.join(root_part5, audio_file1), os.path.join(root_part5, audio_file2)], os.path.join(root_mix_part5, f"{key}.wav"), "diff_time", "PART5"))
        else:
            params_large.append(([os.path.join(root_part5, audio_file1), os.path.join(root_part5, audio_file2)], os.path.join(root_mix_part5, f"{key}.wav"), "diff_time", "PART5"))




    print("Start PART6", flush=True)

    root_mix_part6=f"{root}/PART6/Audio_mixed"
    duration_dict=json.load(open(f"{root}/PART6/duration_dict.json", "r", encoding="utf-8"))

    root_part6=f"{root}/PART6/Audio"
    wav_files = glob(os.path.join(root_part6, '*.wav'), recursive=True)
    wav_files.sort(key=get_key_part6)
    
    for key, value in groupby(wav_files, key=get_key_part6):
        if os.path.exists(os.path.join(root_mix_part6, f"{key}.wav")):
            continue

        try:
            audio_file1, audio_file2 = [os.path.basename(file) for file in list(value)]
        except ValueError as e:
            logging.error(f"Error: {key}")
            continue

        wav_time1    = duration_dict["Audio"][audio_file1]
        wav_time2    = duration_dict["Audio"][audio_file2]

        if abs(wav_time1 - wav_time2)<0.02:
            params.append(([os.path.join(root_part6, audio_file1), os.path.join(root_part6, audio_file2)], os.path.join(root_mix_part6, f"{key}.wav"), "same_time", "PART6"))
        elif abs(wav_time1 - wav_time2)<200:
            params.append(([os.path.join(root_part6, audio_file1), os.path.join(root_part6, audio_file2)], os.path.join(root_mix_part6, f"{key}.wav"), "diff_time", "PART6"))
        else:
            params_large.append(([os.path.join(root_part6, audio_file1), os.path.join(root_part6, audio_file2)], os.path.join(root_mix_part6, f"{key}.wav"), "diff_time", "PART6"))




    shift_dict = json.load(open(f"{root}/mix_shift.json", "r", encoding="utf-8"))
    with Pool(processes=54) as pool:

        # results = list(tqdm(pool.imap_unordered(mix_wav, params), total=len(params)))
        # for file1, file2, shift1, shift2 in results:
        #     shift_dict[file1]=shift1
        #     shift_dict[file2]=shift2
        # json.dump(shift_dict, open(f"{root}/mix_shift.json", "w"), indent=4)

        results = list(tqdm(pool.imap_unordered(mix_wav_large, params_large), total=len(params_large)))
        for file1, file2, shift1, shift2 in results:
            shift_dict[file1]=shift1
            shift_dict[file2]=shift2
        json.dump(shift_dict, open(f"{root}/mix_shift.json", "w"), indent=4)






def main():
    logging.info(f"Start")
    mix()
    logging.info(f"Done")



if __name__=="__main__":
    logging.info(f"Start")
    Fire(main)
    logging.info(f"Done")
