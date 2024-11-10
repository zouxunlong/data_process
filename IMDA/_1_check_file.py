from collections import defaultdict
import os,json
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

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART3"):
        for file in files:
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

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART4"):
        for file in files:
            if not any([re.fullmatch(pattern, file) for pattern in patterns]):
                logging.error(file)


def check_part5_filename():

    patterns = [
        r'^app_\d{4}_\d{4}_phnd_deb-(1|2|3)\.(wav|TextGrid)$',
        r'^app_\d{4}_\d{4}_phnd_(fin|neg|pos)\.(wav|TextGrid)$',
    ]

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART5"):
        for file in files:
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
    ]

    for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6"):
        for file in files:
            if re.fullmatch(patterns[0], file):
                os.rename(os.path.join(root, file), os.path.join("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/Call_Centre_Design_1/Audio", file))
            elif re.fullmatch(patterns[1], file):
                os.rename(os.path.join(root, file), os.path.join("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/Call_Centre_Design_1/Scripts", file))
            elif re.fullmatch(patterns[2], file):
                os.rename(os.path.join(root, file), os.path.join("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/Call_Centre_Design_2/Audio", file))
            elif re.fullmatch(patterns[3], file):
                os.rename(os.path.join(root, file), os.path.join("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/Call_Centre_Design_2/Scripts", file))
            elif re.fullmatch(patterns[4], file):
                os.rename(os.path.join(root, file), os.path.join("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/Call_Centre_Design_3/Audio", file))
            elif re.fullmatch(patterns[5], file):
                os.rename(os.path.join(root, file), os.path.join("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/Call_Centre_Design_3/Scripts", file))
            else:
                logging.error(file)


def get_length(file):
    if file.endswith(".TextGrid"):
        try:
            transcriptions = textgrid.TextGrid.fromFile(file)
            duration = transcriptions.maxTime
        except Exception as e:
            logging.error(f"Error in {file}")
            logging.error(e)
            open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/erorr_files.log", "a", encoding="utf-8").write(file+"\n")
            duration=0

    elif file.endswith(".wav"):
        array, sr = sf.read(file)
        duration=len(array)/sr
    else:
        duration=0
    return {os.path.basename(file):duration}


def get_part3_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict=defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART3"):
            dir= os.path.basename(root)
            params=[(os.path.join(root, file)) for file in files]
            results=list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict, 
                open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART3/duration_dict.json", "w"),
                ensure_ascii=False, indent=4)


def get_part4_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict=defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART4"):
            dir= os.path.basename(root)
            params=[(os.path.join(root, file)) for file in files]
            results=list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict, 
                open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART4/duration_dict.json", "w"),
                ensure_ascii=False, indent=4)


def get_part5_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict=defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART5"):
            dir= os.path.basename(root)
            params=[(os.path.join(root, file)) for file in files]
            results=list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict, 
                open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART5/duration_dict.json", "w"),
                ensure_ascii=False, indent=4)


def get_part6_time(workers=200):
    with Pool(processes=workers) as pool:
        duration_dict=defaultdict(dict)
        for root, dirs, files in os.walk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6"):
            dir= os.path.basename(root)
            params=[(os.path.join(root, file)) for file in files]
            results=list(tqdm(pool.imap(get_length, params), total=len(params)))
            for result in results:
                duration_dict[dir].update(result)
        json.dump(duration_dict, 
                open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/duration_dict.json", "w"),
                ensure_ascii=False, indent=4)


def check_part3_time():
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART3/duration_dict.json", "r", encoding="utf-8"))
    total_same=0
    error_same=0
    for wav, wav_time in duration_dict["Audio_Same_BoundaryMic"].items():
        total_same+=1
        script_time1=duration_dict["Scripts_Same"][wav.split(".")[0]+"-1"+".TextGrid"]
        script_time2=duration_dict["Scripts_Same"][wav.split(".")[0]+"-2"+".TextGrid"]
        if abs(wav_time-script_time1)>0.1 or abs(wav_time-script_time2)>0.1:
            logging.error(f"wav:{wav}, wav_time:{wav_time}, script_time1:{script_time1}, script_time2:{script_time2}")
            error_same+=1

    total_diff=0
    error_diff=0
    for script, script_time in duration_dict["Scripts_Separate"].items():
        total_diff+=1
        if script.split("_")[1] in [key.split(".")[0] for key in duration_dict["Audio_separate_mixed"].keys()]:
            wav_time=duration_dict["Audio_separate_mixed"][script.split("_")[1]+".wav"]
        else:
            wav_time=duration_dict["Audio_Separate_StandingMic"][script.split(".")[0]+".wav"]
        if abs(script_time-wav_time)>0.1:
            logging.error(f"script: {script}, script_time: {script_time}, StandingMic: {wav_time}")
            error_diff+=1
    logging.error(f"PART3:")
    logging.error(f"same_room: {total_same}, error: {error_same}")
    logging.error(f"diff_rrom: {total_diff}, error: {error_diff}")


def check_part4_time():
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART4/duration_dict.json", "r", encoding="utf-8"))
    total=0
    error=0
    for script, script_time in duration_dict["Scripts"].items():
        total+=1
        wav_time=duration_dict["Audio"][script.replace(".TextGrid", ".wav")]
        if abs(wav_time-script_time)>0.1:
            # logging.error(f"script: {script}, script_time/wav_time:{script_time}/{wav_time}")
            error+=1

    logging.error(f"PART4:")
    logging.error(f"total: {total}, error: {error}")


def check_part5_time():
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART5/duration_dict.json", "r", encoding="utf-8"))
    total=0
    error=0
    for script, script_time in duration_dict["Scripts"].items():
        total+=1
        wav_time=duration_dict["Audio"][script.replace(".TextGrid", ".wav")]
        if abs(wav_time-script_time)>0.1:
            # logging.error(f"script:{script}, script_time/wav_time:{script_time}/{wav_time}")
            error+=1

    logging.error(f"PART5:")
    logging.error(f"total: {total}, error: {error}")


def check_part6_time():
    duration_dict=json.load(open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART6/duration_dict.json", "r", encoding="utf-8"))
    total=0
    error=0
    for script, script_time in duration_dict["Scripts"].items():
        total+=1
        wav_time=duration_dict["Audio"][script.replace(".TextGrid", ".wav")]
        if abs(wav_time-script_time)>0.1:
            # logging.error(f"script:{script}, script_time/wav_time:{script_time}/{wav_time}")
            error+=1

    logging.error(f"PART6:")
    logging.error(f"total: {total}, error: {error}")


def try_fix():
    lines=open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/erorr_files.log", "r", encoding="utf-8").readlines()
    for line in lines:
        file=line.strip()
        transcriptions = textgrid.TextGrid.fromFile(file)
        print(file)


def main():
    get_part3_time()
    get_part4_time()
    get_part5_time()
    get_part6_time()


if __name__=="__main__":
    logging.error(f"Script executed starts")
    Fire(main)
    logging.error(f"Script executed ends")
