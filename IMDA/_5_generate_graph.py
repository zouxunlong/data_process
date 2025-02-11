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


def get_key(path):
    if path.split("/")[-3] == "PART3":
        return path.split("/")[-3] + "_" + path.split("/")[-1].split("_")[1]
    else:
        return path.split("/")[-3] + "_" + path.split("/")[-1].split("_")[1] + "_" + path.split("/")[-1].split("_")[-1].split(".")[0]


def wavfile2scriptfile(wav_file):
    part, dir, filename=wav_file.split("/")[-3:]
    script_file = f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/Scripts/{filename.replace('.wav', '.TextGrid')}"
    return script_file


def normalize_sentence(sentence):
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub('(\(|\[|<)[a-zA-Z0-9/\s]*(>|\)|\])', " ", sentence)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def calculate_mix_graph(wav_pair):
    returns = []
    script_pair = {wavfile2scriptfile(key):value for key, value in wav_pair.items()}

    (file1, shift1), (file2, shift2) = script_pair.items()

    transcription_textgrid_1 = textgrid.TextGrid.fromFile(file1)
    transcription_textgrid_2 = textgrid.TextGrid.fromFile(file2)
    transcription_1          = [{"start": interval.minTime, "end": interval.maxTime, "sentence": normalize_sentence(interval.mark)} for interval in transcription_textgrid_1[0] if len(normalize_sentence(interval.mark).split())>3]
    transcription_2          = [{"start": interval.minTime, "end": interval.maxTime, "sentence": normalize_sentence(interval.mark)} for interval in transcription_textgrid_2[0] if len(normalize_sentence(interval.mark).split())>3]
    total_duration_1         = transcription_textgrid_1.maxTime
    total_duration_2         = transcription_textgrid_2.maxTime

    bar1 = [0] * round(total_duration_1)
    bar2 = [0] * round(total_duration_2)

    for interval in transcription_1:
        start_pos = round(interval["start"])
        end_pos   = round(interval["end"])
        for i in range(start_pos, end_pos):
            bar1[i] = 1

    for interval in transcription_2:
        start_pos = round(interval["start"])
        end_pos   = round(interval["end"])
        for i in range(start_pos, end_pos):
            bar2[i] = 1

    bar1 = [0] * max(0, round(shift2/16000)) + bar1
    bar2 = [0] * max(0, round(shift1/16000)) + bar2

    result = [a + b for a, b in zip_longest(bar1, bar2, fillvalue=0)]

    count_of_2 = result.count(2)
    count_of_1 = result.count(1)

    ratio = count_of_2 / (count_of_1 + count_of_2)

    for file, shift in script_pair.items():
        returns.append(f"{file} {ratio:.4f} {shift}")

    mapping    = {0: " ", 1: "|", 2: "█"}
    returns.append(''.join([str(mapping[item]) for item in bar1]))
    returns.append(''.join([str(mapping[item]) for item in bar2]))
    returns.append(''.join([str(mapping[item]) for item in result]))
    return returns


def generate_mix_graph():
    shift_dict = json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_shift.json", "r", encoding="utf-8"))

    params=[]
    for key, value in tqdm(groupby(sorted(shift_dict.items(), key=lambda item: get_key(item[0])), key=lambda item: get_key(item[0]))):
        wav_pair    = {item[0]:item[1] for item in list(value)}
        assert len(wav_pair)==2, f"{key}: {len(wav_pair)}"
        params.append(wav_pair)

    # result = calculate_mix_graph(params[0])

    with Pool(processes=96) as pool:
        resultss = list(tqdm(pool.imap_unordered(calculate_mix_graph, params), total=len(params)))
        with open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_graph_part3.txt", "w") as f_part3, \
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_graph_part4.txt", "w") as f_part4, \
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_graph_part5.txt", "w") as f_part5, \
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_graph_part6.txt", "w") as f_part6, \
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_graph.txt", "w") as f, \
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/mix_graph.json", "w") as f_json:
            shift_with_ratio_dict={}
            
            for  returns in sorted(resultss, key=lambda item: float(item[0].split(" ")[-2].strip())):
                f.write("\n".join(returns) + "\n")
                if returns[0].split("/")[-3] == "PART3":
                    f_part3.write("\n".join(returns) + "\n")
                if returns[0].split("/")[-3] == "PART4":
                    f_part4.write("\n".join(returns) + "\n")
                if returns[0].split("/")[-3] == "PART5":
                    f_part5.write("\n".join(returns) + "\n")
                if returns[0].split("/")[-3] == "PART6":
                    f_part6.write("\n".join(returns) + "\n")

                file, ratio, shift=returns[0].split(" ")
                shift_with_ratio_dict[file] = (float(ratio), int(shift))
                file, ratio, shift=returns[1].split(" ")
                shift_with_ratio_dict[file] = (float(ratio), int(shift))

            f_json.write(json.dumps(shift_with_ratio_dict, indent=4))

def main():
    logging.info(f"Start")
    generate_mix_graph()
    logging.info(f"Done")



if __name__=="__main__":
    logging.info(f"Start")
    Fire(main)
    
    logging.info(f"Done")
