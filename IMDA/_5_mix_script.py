from glob import glob
from itertools import groupby, zip_longest
import os
import json
import numpy as np
from fire import Fire
import logging
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm
import textgrid
import re
import unicodedata


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def normalize_sentence(sentence):
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub('(\(|\[|<)[a-zA-Z0-9/\s]*(>|\)|\])', " ", sentence)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def calculate_overlapping_ratio(bar1, bar2, shift):

    bar1 = [0] * max(0, -shift) + bar1
    bar2 = [0] * max(0, shift) + bar2

    result = [a + b for a, b in zip_longest(bar1, bar2, fillvalue=0)]

    count_of_2 = result.count(2)
    count_of_1 = result.count(1)

    ratio = count_of_2 / (count_of_1 + count_of_2)

    return ratio


def mix_script(params):
    input_files, output_file, part, dir_audio = params
    f_out=f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/mix_shift2.jsonl"

    array1, sr1 = sf.read(f"{dir_audio}/{os.path.basename(input_files[0]).replace('.TextGrid', '.wav')}")
    array2, sr2 = sf.read(f"{dir_audio}/{os.path.basename(input_files[1]).replace('.TextGrid', '.wav')}")

    transcription_textgrid_1 = textgrid.TextGrid.fromFile(input_files[0])
    transcription_textgrid_2 = textgrid.TextGrid.fromFile(input_files[1])

    total_duration_1 = transcription_textgrid_1.maxTime
    total_duration_2 = transcription_textgrid_2.maxTime

    transcription_1 = [{"start": interval.minTime, "end": interval.maxTime, "sentence": normalize_sentence(interval.mark)} for interval in transcription_textgrid_1[0] if len(normalize_sentence(interval.mark).split())>3]
    transcription_2 = [{"start": interval.minTime, "end": interval.maxTime, "sentence": normalize_sentence(interval.mark)} for interval in transcription_textgrid_2[0] if len(normalize_sentence(interval.mark).split())>3]
    length_diff     = total_duration_1 - total_duration_2
    max_ratio       = 1

    bar1 = [0] * round(total_duration_1*10)
    bar2 = [0] * round(total_duration_2*10)

    for interval in transcription_1:
        start_pos = round(interval["start"]*10)
        end_pos   = round(interval["end"]*10)
        for i in range(start_pos, end_pos):
            bar1[i] = 1

    for interval in transcription_2:
        start_pos = round(interval["start"]*10)
        end_pos   = round(interval["end"]*10)
        for i in range(start_pos, end_pos):
            bar2[i] = 1

    shift_range = range(round(min(-30, -30 + length_diff) * 10), round(max(30, length_diff + 30) * 10))

    for shift in shift_range:
        ratio = calculate_overlapping_ratio(bar1, bar2, shift)
        if ratio < max_ratio:
            max_ratio     = ratio
            optimal_shift = shift

    start1 = max(0, optimal_shift*1600)
    end1   = min(len(array1), len(array2) + optimal_shift*1600)
    start2 = max(0, - optimal_shift*1600)
    end2   = min(len(array1) - optimal_shift*1600, len(array2))

    mixed_array = array1[start1:end1] + array2[start2:end2]
    sf.write(output_file, mixed_array, 16000)

    open(f_out, "a").write(json.dumps({input_files[0]: start1}) + "\n")
    open(f_out, "a").write(json.dumps({input_files[1]: start2}) + "\n")

    return (input_files[0], input_files[1], start1, start2)


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

    params = []
    root = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw"

    for part in ["PART3", "PART4", "PART5", "PART6"]:
        os.makedirs(f"{root}/{part}/Audio_mixed2", exist_ok=True)

        print(f"Start {part}", flush=True)
        dir_mix2   = f"{root}/{part}/Audio_mixed2"
        dir_script = f"{root}/{part}/Scripts"
        dir_audio  = f"{root}/{part}/Audio"
        files      = glob(os.path.join(dir_script, '*.TextGrid'), recursive=True)
        files.sort(key = get_key)
        for key, value in groupby(files, key = get_key):
            # if os.path.exists(os.path.join(dir_mix2, f"{key}.wav")):
            #     continue
            try:
                file1, file2 = [os.path.basename(file) for file in list(value)]
            except ValueError as e:
                logging.error(f"{e}: {key}")
                continue
            params.append(([os.path.join(dir_script, file1), os.path.join(dir_script, file2)], os.path.join(dir_mix2, f"{key}.wav"), part, dir_audio))
    
    
    shift_dict = json.load(open(f"{root}/mix_shift2.json", "r", encoding="utf-8"))
    with Pool(processes=192) as pool:
        results = list(tqdm(pool.imap_unordered(mix_script, params), total=len(params)))
        for file1, file2, shift1, shift2 in results:
            shift_dict[file1]=shift1
            shift_dict[file2]=shift2
        json.dump(shift_dict, open(f"{root}/mix_shift2.json", "w"), indent=4)


def main():
    logging.info(f"Start")
    mix()
    logging.info(f"Done")


if __name__=="__main__":
    logging.info(f"Start")
    Fire(main)
    logging.info(f"Done")
