from collections import defaultdict
import os
import json
import re
import textgrid
from fire import Fire
import logging
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def check_part3_filename():

    patterns = [
        r'^\d{4}-\d\.TextGrid$',
        r'^conf_\d{4}_\d{4}_\d{8}\.TextGrid$',
        r'^\d{4}-\d\.wav$',
        r'^conf_\d{4}_\d{4}_\d{8}\.wav$',
        r'^\d{4}\.wav$'
    ]

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3"):
        for file in files:
            if "NFA_output" in root:
                continue
            if not any([re.fullmatch(pattern, file) for pattern in patterns]):
                logging.error(file)

def check_part4_filename():

    patterns = [
        r'^sur_\d{4}_\d{4}_phnd_cs-(chn|tml|mly)\.wav$',
        r'^sur_\d{4}_\d{4}_phnd_cs-(chn|tml|mly)\.TextGrid$',
        r'^sur_\d{4}_\d{4}_phns_cs-(chn|tml|mly)\.wav$',
        r'^sur_\d{4}_\d{4}_phns_cs-(chn|tml|mly)\.TextGrid$',
        r'^\d{4}\.wav$'
    ]

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4"):
        for file in files:
            if "NFA_output" in root:
                continue
            if not any([re.fullmatch(pattern, file) for pattern in patterns]):
                logging.error(file)

def check_part5_filename():

    patterns = [
        r'^app_\d{4}_\d{4}_phnd_deb-(1|2|3)\.(wav|TextGrid)$',
        r'^app_\d{4}_\d{4}_phnd_(fin|neg|pos)\.(wav|TextGrid)$',
        r'^\d{4}_(neg|pos|fin|deb-1|deb-2|deb-3)\.wav$'
    ]

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5"):
        for file in files:
            if "NFA_output" in root:
                continue
            if not any([re.fullmatch(pattern, file) for pattern in patterns]):
                logging.error(file)

def check_part6_filename():

    patterns = [
        r'^app_\d{4}_\d{4}_phnd_cc-(hol|hot|res)\.wav$',
        r'^app_\d{4}_\d{4}_phnd_cc-(hol|hot|res)\.TextGrid$',
        r'^app_\d{4}_\d{4}_phnd_cc-(bnk|ins|tel)\.wav$',
        r'^app_\d{4}_\d{4}_phnd_cc-(bnk|ins|tel)\.TextGrid$',
        r'^app_\d{4}_\d{4}_phnd_cc-(hdb|moe|msf)\.wav$',
        r'^app_\d{4}_\d{4}_phnd_cc-(hdb|moe|msf)\.TextGrid$',
        r'^\d{4}_(hot|hol|res|msf|moe|tel|ins|bnk|hdb)\.wav$'
    ]

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6"):
        for file in files:
            if "NFA_output" in root:
                continue
            if not any([re.fullmatch(pattern, file) for pattern in patterns]):
                logging.error(file)


def get_length(file):
    if file.endswith(".TextGrid"):
        try:
            transcriptions = textgrid.TextGrid.fromFile(file)
            duration = transcriptions.maxTime
        except Exception as e:
            logging.error(f"Error in {file}")
            logging.error(e)
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/erorr_files.log",
                 "a", encoding="utf-8").write(file+"\n")
            duration = 0

    elif file.endswith(".wav"):
        array, sr = sf.read(file)
        duration = len(array)/sr
    else:
        duration = 0
    return {os.path.basename(file): duration}


def get_part3_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict = defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3"):
            dir = os.path.basename(root)
            params = [(os.path.join(root, file)) for file in files if file.endswith(".wav") or file.endswith(".TextGrid")]
            results = list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict,
                  open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3/duration_dict.json", "w"),
                  ensure_ascii=False, indent=4)

def get_part4_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict = defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4"):
            dir = os.path.basename(root)
            params = [(os.path.join(root, file)) for file in files if file.endswith(".wav") or file.endswith(".TextGrid")]
            results = list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict,
                  open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/duration_dict.json", "w"),
                  ensure_ascii=False, indent=4)

def get_part5_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict = defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5"):
            dir = os.path.basename(root)
            params = [(os.path.join(root, file)) for file in files if file.endswith(".wav") or file.endswith(".TextGrid")]
            results = list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict,
                  open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5/duration_dict.json", "w"),
                  ensure_ascii=False, indent=4)

def get_part6_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict = defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6"):
            dir = os.path.basename(root)
            params = [(os.path.join(root, file)) for file in files if file.endswith(".wav") or file.endswith(".TextGrid")]
            results = list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict,
                  open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/duration_dict.json", "w"),
                  ensure_ascii=False, indent=4)


def check_part3_time():

    wav_script_pairs = []
    root             = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3"
    duration_dict    = json.load(open(f"{root}/duration_dict.json", "r", encoding="utf-8"))

    for script, script_time in duration_dict["Scripts_Same"].items():
        wav_time    = duration_dict["Audio_Same_CloseMic"][script.replace(".TextGrid", ".wav")]
        script_file = os.path.join(f"{root}/Scripts_Same", script)
        wav_file    = os.path.join(f"{root}/Audio_Same_CloseMic", script.replace(".TextGrid", ".wav"))
        wav_script_pairs.append({"script_time": script_time, 
                                  "wav_time": wav_time, 
                                  "script_file": script_file, 
                                  "wav_file": wav_file})

    for script, script_time in duration_dict["Scripts_Separate"].items():
        wav_time    = duration_dict["Audio_Separate"][script.replace(".TextGrid", ".wav")]
        script_file = os.path.join(f"{root}/Scripts_Separate", script)
        wav_file    = os.path.join(f"{root}/Audio_Separate", script.replace(".TextGrid", ".wav"))
        wav_script_pairs.append({"script_time": script_time, 
                                  "wav_time": wav_time, 
                                  "script_file": script_file, 
                                  "wav_file": wav_file})


    with open(f"{root}/wav_script_pairs.jsonl", "w", encoding="utf-8") as f:
        for wav_script_pair in wav_script_pairs:
            f.write(json.dumps(wav_script_pair, ensure_ascii=False)+"\n")

    logging.info(f"PART3:")
    logging.info(f"total_wav_script_pairs: {len(wav_script_pairs)}")


def check_part4_time():

    wav_script_pairs = []
    root          = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4"
    duration_dict = json.load(open(f"{root}/duration_dict.json", "r", encoding="utf-8"))

    for script, script_time in duration_dict["Scripts"].items():
        wav_time    = duration_dict["Audio"][script.replace(".TextGrid", ".wav")]
        script_file = os.path.join(f"{root}/Scripts", script)
        wav_file    = os.path.join(f"{root}/Audio", script.replace(".TextGrid", ".wav"))
        wav_script_pairs.append({"script_time": script_time, 
                                 "wav_time": wav_time, 
                                 "script_file": script_file, 
                                 "wav_file": wav_file})

    with open(f"{root}/wav_script_pairs.jsonl", "w", encoding="utf-8") as f:
        for wav_script_pair in wav_script_pairs:
            f.write(json.dumps(wav_script_pair, ensure_ascii=False)+"\n")

    logging.error(f"PART4:")
    logging.info(f"total_wav_script_pairs: {len(wav_script_pairs)}")


def check_part5_time():

    wav_script_pairs = []
    root          = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5"
    duration_dict = json.load(open(f"{root}/duration_dict.json", "r", encoding="utf-8"))

    for script, script_time in duration_dict["Scripts"].items():
        wav_time    = duration_dict["Audio"][script.replace(".TextGrid", ".wav")]
        script_file = os.path.join(f"{root}/Scripts", script)
        wav_file    = os.path.join(f"{root}/Audio", script.replace(".TextGrid", ".wav"))
        wav_script_pairs.append({"script_time": script_time, 
                                 "wav_time": wav_time, 
                                 "script_file": script_file, 
                                 "wav_file": wav_file})

    with open(f"{root}/wav_script_pairs.jsonl", "w", encoding="utf-8") as f:
        for wav_script_pair in wav_script_pairs:
            f.write(json.dumps(wav_script_pair, ensure_ascii=False)+"\n")

    logging.error(f"PART5:")
    logging.info(f"total_wav_script_pairs: {len(wav_script_pairs)}")


def check_part6_time():

    wav_script_pairs = []
    root          = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6"
    duration_dict = json.load(open(f"{root}/duration_dict.json", "r", encoding="utf-8"))

    for script, script_time in duration_dict["Scripts"].items():
        wav_time    = duration_dict["Audio"][script.replace(".TextGrid", ".wav")]
        script_file = os.path.join(f"{root}/Scripts", script)
        wav_file    = os.path.join(f"{root}/Audio", script.replace(".TextGrid", ".wav"))
        wav_script_pairs.append({"script_time": script_time, 
                                 "wav_time": wav_time, 
                                 "script_file": script_file, 
                                 "wav_file": wav_file})

    with open(f"{root}/wav_script_pairs.jsonl", "w", encoding="utf-8") as f:
        for wav_script_pair in wav_script_pairs:
            f.write(json.dumps(wav_script_pair, ensure_ascii=False)+"\n")

    logging.error(f"PART6:")
    logging.info(f"total_wav_script_pairs: {len(wav_script_pairs)}")


def check(textgrid_file):
    try:
        transcriptions = textgrid.TextGrid.fromFile(textgrid_file)
        assert len(transcriptions) == 1
    except Exception as e:
        logging.error(f"Error in {textgrid_file}")
        open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/erorr_files.log",
                "a", encoding="utf-8").write(textgrid_file+"\n")


def check_textgrid():
    
    with Pool(processes=32) as pool:
        params=[]

        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw"): 
            for file in files:
                if file.endswith(".TextGrid"):
                    params.append(os.path.join(root, file))

        results = list(tqdm(pool.imap(check, params), total=len(params)))


def check_part3_mis_match():
    root             = "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART3"
    duration_dict    = json.load(open(f"{root}/duration_dict.json", "r", encoding="utf-8"))

    for script, script_time in duration_dict["Scripts_Same"].items():
        wav_time    = duration_dict["Audio_Same_CloseMic"][script.replace(".TextGrid", ".wav")]
        if abs(script_time-wav_time) > 0.01:
            print(script, format(script_time-wav_time, ".2f"))  

    for script, script_time in duration_dict["Scripts_Separate"].items():
        wav_time    = duration_dict["Audio_Separate"][script.replace(".TextGrid", ".wav")]
        if abs(script_time-wav_time) > 0.01:
            print(script, format(script_time-wav_time, ".2f"))  

    for audio, audio_time in duration_dict["Audio_Same"].items():
        wav_time_1 = duration_dict["Audio_Same_CloseMic"][audio.replace(".wav", "-1.wav")]
        wav_time_2 = duration_dict["Audio_Same_CloseMic"][audio.replace(".wav", "-2.wav")]
        if abs(wav_time_1-wav_time_2) > 0.01 or abs(wav_time_1-audio_time) > 0.01 or abs(wav_time_2-audio_time) > 0.01:
            print(audio, format(wav_time_1-wav_time_2, ".2f"))


def main():
    
    check_part3_time()


if __name__ == "__main__":
    logging.error(f"Script executed starts")
    Fire(main)
    logging.error(f"Script executed ends")
