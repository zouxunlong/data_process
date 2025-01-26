import json
import pandas as pd
from fire import Fire
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def fetch_speaker_metadata_part1():

    speaker_dict = {}
    df = pd.read_excel("/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/Part_1_Speaker_Metadata.xlsx", dtype=str)
    for i, values in enumerate(tqdm(df.fillna('Unknown').loc[0:].values)):
        speaker_id   = values[0].strip().capitalize()
        part1_id     = values[1].strip().capitalize()
        part2_id     = values[2].strip().capitalize()
        gender       = values[3].strip().capitalize()
        ethnic_group = values[4].strip().capitalize()
        device_c0    = values[5].strip().capitalize()
        device_c1    = values[6].strip().capitalize()
        device_c2    = values[7].strip().capitalize()

        speaker_dict[part1_id] = {
            "speaker_id"  : speaker_id,
            "part1_id"    : part1_id,
            "part2_id"    : part2_id,
            "gender"      : gender,
            "ethnic_group": ethnic_group,
            "device_c0"   : device_c0,
            "device_c1"   : device_c1,
            "device_c2"   : device_c2,
        }
    logging.info('PART1 Num of speakers = {}'.format(len(speaker_dict)))
    json.dump(speaker_dict, open("speaker_metadata_part1.json", "w"), ensure_ascii=False, indent=4)
    return speaker_dict


def fetch_speaker_metadata_part2():

    speaker_dict = {}
    df = pd.read_excel("/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/Part_2_Speaker_Metadata.xlsx", dtype=str)
    for i, values in enumerate(tqdm(df.fillna('Unknown').loc[0:].values)):
        speaker_id   = values[0].strip().capitalize()
        part1_id     = values[1].strip().capitalize()
        part2_id     = values[2].strip().capitalize()
        gender       = values[3].strip().capitalize()
        ethnic_group = values[4].strip().capitalize()
        device_c0    = values[5].strip().capitalize()
        device_c1    = values[6].strip().capitalize()
        device_c2    = values[7].strip().capitalize()
    
        speaker_dict[part2_id] = {
            "speaker_id"  : speaker_id,
            "part1_id"    : part1_id,
            "part2_id"    : part2_id,
            "gender"      : gender,
            "ethnic_group": ethnic_group,
            "device_c0"   : device_c0,
            "device_c1"   : device_c1,
            "device_c2"   : device_c2,
        }
    logging.info('PART2 Num of speakers = {}'.format(len(speaker_dict)))
    json.dump(speaker_dict, open("speaker_metadata_part2.json", "w"), ensure_ascii=False, indent=4)
    return speaker_dict


def fetch_speaker_metadata_part3():

    speaker_dict = {"same_room": {}, "separate_room": {}}
    df_dict = pd.read_excel("/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/Part_3_Speaker_Metadata.xlsx", sheet_name=['Same Room', 'Separate Room'], dtype=str)
    for i, values in enumerate(tqdm(df_dict['Same Room'].fillna('Unknown').loc[0:].values)):
        speaker_id           = values[0].strip().capitalize()
        age                  = values[2].strip().capitalize()
        gender               = values[1].strip().capitalize()
        education_level      = values[4].strip().capitalize()
        occupation           = values[5].strip().capitalize()
        ethnic_group         = values[3].strip().capitalize()
        first_language       = values[7].strip().capitalize()
        spoken_language      = values[6].strip().capitalize()
        partner_id           = values[8].strip().capitalize()
        partner_relationship = values[9].strip().capitalize()
        speaker_dict["same_room"][speaker_id]={
            "speaker_id"          : speaker_id,
            "age"                 : age,
            "gender"              : gender,
            "ethnic_group"        : ethnic_group,
            "education_level"     : education_level,
            "occupation"          : occupation,
            "first_language"      : first_language,
            "spoken_language"     : spoken_language,
            "partner_id"          : partner_id,
            "partner_relationship": partner_relationship,
        }
    logging.info('PART3 Num of speakers in same room = {}'.format(len(speaker_dict["same_room"])))

    for i, values in enumerate(tqdm(df_dict['Separate Room'].fillna('Unknown').loc[0:].values)):
        conference_id        = values[0].strip().capitalize()
        speaker_id           = values[1].strip().capitalize()
        gender               = values[2].strip().capitalize()
        age                  = values[3].strip().capitalize()
        ethnic_group         = values[4].strip().capitalize()
        education_level      = values[5].strip().capitalize()
        occupation           = values[6].strip().capitalize()
        spoken_language      = values[7].strip().capitalize()
        first_language       = values[8].strip().capitalize()
        partner_id           = values[9].strip().capitalize()
        partner_relationship = values[10].strip().capitalize()
        speaker_dict["separate_room"][speaker_id]={
            "conference_id"       : conference_id,
            "speaker_id"          : speaker_id,
            "age"                 : age,
            "gender"              : gender,
            "ethnic_group"        : ethnic_group,
            "education_level"     : education_level,
            "occupation"          : occupation,
            "first_language"      : first_language,
            "spoken_language"     : spoken_language,
            "partner_id"          : partner_id,
            "partner_relationship": partner_relationship,
        }
    logging.info('PART3 Num of speakers in seperate room = {}'.format(len(speaker_dict["separate_room"])))
    json.dump(speaker_dict, open("speaker_metadata_part3.json", "w"), ensure_ascii=False, indent=4)
    return speaker_dict


def fetch_speaker_metadata_part4():

    speaker_dict = {}
    
    df = pd.read_excel("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/Part_4_Speaker_Metadata.xlsx", dtype=str)
    for i, values in enumerate(tqdm(df.fillna('Unknown').loc[0:].values)):
        session_id               = values[0].strip().capitalize()
        speaker_id               = values[1].strip().capitalize()
        partner_relationship     = values[2].strip().capitalize()
        age                      = values[3].strip().capitalize()
        gender                   = values[4].strip().capitalize()
        ethnic_group             = values[5].strip().capitalize()
        education_level          = values[6].strip().capitalize()
        occupation               = values[7].strip().capitalize()
        first_language           = values[8].strip().capitalize()
        dominant_language        = values[9].strip().capitalize()
        spoken_language          = values[10].strip().capitalize()
        partner_id               = values[11].strip().capitalize()
        speaker_dict[speaker_id] = {
            "session_id"          : session_id,
            "speaker_id"          : speaker_id,
            "partner_id"          : partner_id,
            "age"                 : age,
            "gender"              : gender,
            "ethnic_group"        : ethnic_group,
            "education_level"     : education_level,
            "occupation"          : occupation,
            "first_language"      : first_language,
            "dominant_language"   : dominant_language,
            "spoken_language"     : spoken_language,
            "partner_relationship": partner_relationship,
        }
    logging.info('PART4 Num of speakers = {}'.format(len(speaker_dict)))
    json.dump(speaker_dict, open("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/speaker_metadata_part4.json", "w"), ensure_ascii=False, indent=4)
    return speaker_dict


def fetch_speaker_metadata_part5():

    speaker_dict = {}
    df_dict = pd.read_excel("/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/Part_5_Speaker_Metadata.xlsx", sheet_name=['Main', 'Finance', "Debate"], dtype=str)
    
    for i, values in enumerate(tqdm(df_dict['Main'].fillna('Unknown').loc[0:].values)):
        speaker_id           = values[1].strip().capitalize()
        age                  = values[3].strip().capitalize()
        gender               = values[4].strip().capitalize()
        education_level      = values[5].strip().capitalize()
        occupation           = values[6].strip().capitalize()
        ethnic_group         = values[7].strip().capitalize()
        first_language       = values[8].strip().capitalize()
        dominant_language    = values[9].strip().capitalize()
        spoken_language      = values[10].strip().capitalize()
        partner_relationship = values[11].strip().capitalize()
        speaker_dict[speaker_id]={
            "speaker_id"          : speaker_id,
            "age"                 : age,
            "gender"              : gender,
            "ethnic_group"        : ethnic_group,
            "education_level"     : education_level,
            "occupation"          : occupation,
            "first_language"      : first_language,
            "dominant_language"   : dominant_language,
            "spoken_language"     : spoken_language,
            "partner_relationship": partner_relationship,
        }
    logging.info('PART5 Num of speakers in main = {}'.format(len(speaker_dict)))

    for i, values in enumerate(tqdm(df_dict['Finance'].fillna('Unknown').loc[0:].values)):
        speaker_id           = values[2].strip().capitalize()
        age                  = values[3].strip().capitalize()
        gender               = values[4].strip().capitalize()
        education_level      = values[5].strip().capitalize()
        occupation           = values[6].strip().capitalize()
        ethnic_group         = values[7].strip().capitalize()
        first_language       = values[8].strip().capitalize()
        dominant_language    = values[9].strip().capitalize()
        spoken_language      = values[10].strip().capitalize()
        partner_relationship = values[11].strip().capitalize()
        speaker_dict[speaker_id]={
            "speaker_id"          : speaker_id,
            "age"                 : age,
            "gender"              : gender,
            "ethnic_group"        : ethnic_group,
            "education_level"     : education_level,
            "occupation"          : occupation,
            "first_language"      : first_language,
            "dominant_language"   : dominant_language,
            "spoken_language"     : spoken_language,
            "partner_relationship": partner_relationship,
        }
    logging.info('PART5 Num of speakers in finance = {}'.format(len(speaker_dict)))

    for i, values in enumerate(tqdm(df_dict['Debate'].fillna('Unknown').loc[0:].values)):
        speaker_id           = values[2].strip().capitalize()
        age                  = values[3].strip().capitalize()
        gender               = values[4].strip().capitalize()
        education_level      = values[5].strip().capitalize()
        occupation           = values[6].strip().capitalize()
        ethnic_group         = values[7].strip().capitalize()
        first_language       = values[8].strip().capitalize()
        dominant_language    = values[9].strip().capitalize()
        spoken_language      = values[10].strip().capitalize()
        partner_relationship = values[11].strip().capitalize()
        speaker_dict[speaker_id]={
            "speaker_id"          : speaker_id,
            "age"                 : age,
            "gender"              : gender,
            "ethnic_group"        : ethnic_group,
            "education_level"     : education_level,
            "occupation"          : occupation,
            "first_language"      : first_language,
            "dominant_language"   : dominant_language,
            "spoken_language"     : spoken_language,
            "partner_relationship": partner_relationship,
        }
    logging.info('PART5 Num of speakers in debate = {}'.format(len(speaker_dict)))

    json.dump(speaker_dict, open("speaker_metadata_part5.json", "w"), ensure_ascii=False, indent=4)
    return speaker_dict


def main():

    fetch_speaker_metadata_part3()



if __name__=="__main__":
    logging.info(f"Script executed starts")
    Fire(main)
    logging.info(f"Script executed ends")
