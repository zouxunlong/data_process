from collections import defaultdict
from glob import glob
from itertools import groupby
import traceback
from datasets import load_from_disk, concatenate_datasets
import random
import re
from fire import Fire
import os


instructions_GR_first_qa = ([
    "What is the gender of the first speaker in the dialogue?",
    "What is the gender of the speaker who initiates the conversation in the given dialogue?",
    "What is the gender of the speaker who starts the conversation?",
    "What is the speaker's gender at the beginning of the dialogue?",
    "What is the voice characteristic of the first speaker, male or female, in the dialogue?",
    "Does the speaker at the start of the dialogue have a male or female voice?",
    "In the dialogue, what is the gender of the person speaking at the beginning?",
    "What is the gender of the speaker who initiates the conversation in the dialogue?",
    "What is the gender of the first speaker?",
    "What gender is the individual who starts speaking?",
    "Which gender does the first speaker have?",
    "What is the gender of the initial speaker in the conversation?"
], "The gender of the first speaker is {first_gender}.")

instructions_GR_second_qa = ([
    "What is the gender of the second speaker in the dialogue?",
    "In the dialogue, what is the gender of the person speaking secondly?",
    "What is the second speaker's gender in the conversation?",
    "What is the voice characteristic of the second speaker, male or female, in the dialogue?",
    "What is the gender of the second speaker?",
    "Which gender does the second speaker have?",
    "What is the gender of the second speaker in the conversation?"
], "The gender of the second speaker is {second_gender}.")

instructions_GR_all_qa = ([
    "What is the gender of the speakers in the dialogue?",
    "What is the gender of the two people speaking in the dialogue?",
    "Identify the gender of the speakers who are talking",
    "Can you determine the gender of the speakers speaking",
], "The gender of the speakers in the dialogue is a {first_gender} and a {second_gender}.")

instructions_GR_all_only_qa = ([
    "What is the gender of the speakers in the dialogue?",
    "What is the gender of the two people speaking in the dialogue?",
    "Identify the gender of the speakers who are talking.",
    "Can you determine the gender of the speakers speaking?",
], "The gender of the only speaker in the audio is {only_gender}.")


instructions_NR_first_qa = ([
    "Based on the accent of the speaker(s), what is the ethnic group of the first speaker in the dialogue?",
    "What is the ethnic origin of the first speaker in the dialogue?",
    "Based on the audio recording, what is the nationality of the first speaker?",
    "Identify the ethnicity of the speaker who starts speaking first in the audio recording.",
    "Based on the accent in the audio, what is the ethnic group of the first speaker in the dialogue?"
], "Based on the accent, the ethnic group of the first speaker is {first_ethnic_group}.")

instructions_NR_second_qa = ([
    "Based on the accent of the speaker(s), what is the ethnic group of the second speaker in the dialogue?",
    "What is the ethnic origin of the second speaker in the dialogue?",
    "Based on the audio recording, what is the nationality of the second speaker?",
    "Identify the ethnicity of the speaker who starts speaking second in the audio recording.",
    "Based on the accent in the audio, what is the ethnic group of the second speaker in the dialogue?"
], "Based on the accent, the ethnic group of the second speaker is {second_ethnic_group}.")

instructions_NR_all_qa = ([
    "Based on the accent of the speaker(s), what are the ethnic groups of the speakers?",
    "What is the ethnic origin of the speakers in the dialogue?",
    "Based on the audio recording, what is the nationality of the second speaker?",
    "Identify the ethnicity of the speakers in the audio recording.",
    "Based on the audio recording, what is the ethnic group of the speakers in the dialogue?",
    "Based on the accent, what is the ethnic group of the speakers in the dialogue?",
], "Based on the accent, the ethnic group of the speakers in the audio record is one {first_ethnic_group} and one {second_ethnic_group}.")

instructions_NR_all_only_qa = ([
    "Based on the accent of the speaker(s), what are the ethnic groups of the speakers?",
    "What is the ethnic origin of the speakers in the dialogue?",
    "Based on the audio recording, what is the nationality of the second speaker?",
    "Identify the ethnicity of the speakers in the audio recording.",
    "Based on the audio recording, what is the ethnic group of the speakers in the dialogue?",
    "Based on the accent, what is the ethnic group of the speakers in the dialogue?",
], "Based on the accent, the ethnic group of the only speaker in the audio record is {only_ethnic_group}.")


instructions_MIX_GR_NR_first_qa = ([
    "Could you infer from the audio, what is the gender and ethnic group of the first speaker in the dialogue?",
    "Based on the audio clip, what is the gender and ethnic group of the first speaker in the dialogue?",
    "What is the gender and ethnic group of the first speaker in the dialogue?",
], "Based on the accent, the first speaker is a {first_ethnic_group} {first_gender}.")

instructions_MIX_GR_NR_second_qa = ([
    "Could you infer from the audio, what is the gender and ethnic group of the second speaker in the dialogue?",
    "Based on the audio clip, what is the gender and ethnic group of the second speaker in the dialogue?",
    "What is the gender and ethnic group of the second speaker in the dialogue?",
], "Based on the accent, the second speaker is a {second_ethnic_group} {second_gender}.")

instructions_MIX_GR_NR_all_qa = ([
    "What is the gender and ethnic group of the speakers in the dialogue?",
    "What are the genders and ethnic groups of the individuals speaking in the dialogue?",
    "What is the gender and ethnic composition of the speakers in the conversation?",
    "What is the gender and ethnic group composition of the conversation participants?",
    "Can you identify the gender and ethnic group of the speakers in this dialogue?",
], "The speakers in the audio record is one {first_ethnic_group} {first_gender} and one {second_ethnic_group} {second_gender}.")

instructions_MIX_GR_NR_all_only_qa = ([
    "What is the gender and ethnic group of the speakers in the dialogue?",
    "What are the genders and ethnic groups of the individuals speaking in the dialogue?",
    "What is the gender and ethnic composition of the speakers in the conversation?",
    "What is the gender and ethnic group composition of the conversation participants?",
    "Can you identify the gender and ethnic group of the speakers in this dialogue?",
], "The speaker in the audio record is {only_ethnic_group} {only_gender}")

instructions_MIX_COUNT_GR_qa = ([
    "How many speakers are there in the audio clip? What are their genders?",
    "How many voices are present in the recording, and what is the gender breakdown?",
    "What is the number of speakers, and what are their corresponding genders?",
    "Can you tell me how many speakers are in the audio clip, and what are their genders?",
    "How many individuals are speaking in the audio, and can you identify their genders?",
    "What can you say about the speakers in the audio recording in terms of number and gender?",
    "How many speakers are there in the audio clip? What are their genders?",
], "There are two speakers in the audio clip, one {first_gender} and one {second_gender}.")

instructions_MIX_COUNT_GR_only_qa = ([
    "How many speakers are there in the audio clip? What are their genders?",
    "How many voices are present in the recording, and what is the gender breakdown?",
    "What is the number of speakers, and what are their corresponding genders?",
    "Can you tell me how many speakers are in the audio clip, and what are their genders?",
    "How many individuals are speaking in the audio, and can you identify their genders?",
    "What can you say about the speakers in the audio recording in terms of number and gender?",
    "How many speakers are there in the audio clip? What are their genders?",
], "There is one {only_gender} speaker in the audio clip.")

instructions_MIX_COUNT_NR_qa = ([
    "How many speakers are there in the audio clip? What are their ethnic groups?",
    "How many individuals are speaking in the audio recording, and what are their ethnic backgrounds?",
    "What is the total number of voices in the audio clip, and what are their respective ethnicities?",
    "In the audio recording, what is the count of speakers, and what are their racial origins?",
    "In the audio clip, how many people are speaking, and what are their ethnic identities?",
    "What is the speaker count in the audio recording, and what are their ethnic affiliations?",
], "There are two speakers in the audio clip, one is {first_ethnic_group} and the other is {second_ethnic_group}.")

instructions_MIX_COUNT_NR_only_qa = ([
    "How many speakers are there in the audio clip? What are their ethnic groups?",
    "How many individuals are speaking in the audio recording, and what are their ethnic backgrounds?",
    "What is the total number of voices in the audio clip, and what are their respective ethnicities?",
    "In the audio recording, what is the count of speakers, and what are their racial origins?",
    "In the audio clip, how many people are speaking, and what are their ethnic identities?",
    "What is the speaker count in the audio recording, and what are their ethnic affiliations?",
], "There is one speaker in the audio record, and it is {only_ethnic_group} based on the accent.")

instructions_MIX_COUNT_NR_GR_qa = ([
    "How many speakers are present in the audio recording, and what are their ethnic backgrounds and genders?",
    "How many individuals are speaking? What can be inferred about the speakers in terms of gender and ethnic group based on their voices?",
    "In the audio file, how many speakers are there? What are the characteristics of the speakers, including ethnicity and gender?",
    "What is the breakdown of speakers in the audio recording in terms of ethnicity, gender and number of speakers?",
], "There is one speaker of {first_ethnic_group} ethnicity and {first_gender} gender, and another speaker of {second_ethnic_group} ethnicity and {second_gender} gender.")

instructions_MIX_COUNT_NR_GR_only_qa = ([
    "How many speakers are present in the audio recording, and what are their ethnic backgrounds and genders?",
    "How many individuals are speaking? What can be inferred about the speakers in terms of gender and ethnic group based on their voices?",
    "In the audio file, how many speakers are there? What are the characteristics of the speakers, including ethnicity and gender?",
    "What is the breakdown of speakers in the audio recording in terms of ethnicity, gender and number of speakers?",
], "There is one speaker of {only_ethnic_group} ethnicity and {only_gender} gender.")


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def mapping_GR(batch, index):
    newbatch={"instruction":[], "answer":[]}
    try:
        for i in range(len(batch["answer"])):
            transcription=batch["answer"][i]["text"]

            if re.search("\n", transcription): num_of_speaker=2
            else: num_of_speaker=1

            first_speaker=batch["other_attributes"][i]["speaker1"]
            second_speaker=batch["other_attributes"][i]["speaker2"]


            first_gender="female" if first_speaker["gender"] in ["F", "Female"] else "male"
            second_gender="female" if second_speaker["gender"] in ["F", "Female"] else "male"

            if index==0:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_first_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_first_qa[1].format(first_gender=first_gender),"audio": None})

            if index==1:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_first_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_first_qa[1].format(first_gender=first_gender),"audio": None})

            if index==2 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_second_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_second_qa[1].format(second_gender=second_gender),"audio": None})

            if index==3 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_second_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_second_qa[1].format(second_gender=second_gender),"audio": None})

            if index==4 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_all_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_all_qa[1].format(first_gender=first_gender, second_gender=second_gender),"audio": None})


            if index==2 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_all_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_all_only_qa[1].format(only_gender=first_gender),"audio": None})

            if index==3 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_all_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_all_only_qa[1].format(only_gender=first_gender),"audio": None})

            if index==4 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_GR_first_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_GR_first_qa[1].format(first_gender=first_gender),"audio": None})

        return newbatch
    except:
        print("exception found", flush=True)
        print(traceback.format_exc(), flush=True)
        return newbatch


def mapping_NR(batch, index):
    newbatch={"instruction":[], "answer":[]}
    try:
        for i in range(len(batch["answer"])):
            transcription=batch["answer"][i]["text"]

            if re.search("\n", transcription): num_of_speaker=2
            else: num_of_speaker=1

            first_speaker=batch["other_attributes"][i]["speaker1"]
            second_speaker=batch["other_attributes"][i]["speaker2"]

            first_ethnic_group=first_speaker["ethnic_group"]
            second_ethnic_group=second_speaker["ethnic_group"]


            if index==0:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_first_qa[0]), "audio": None})
                newbatch["answer"].append({"text": instructions_NR_first_qa[1].format(first_ethnic_group=first_ethnic_group), "audio": None})

            if index==1:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_first_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_first_qa[1].format(first_ethnic_group=first_ethnic_group),"audio": None})

            if index==2 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_second_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_second_qa[1].format(second_ethnic_group=second_ethnic_group),"audio": None})

            if index==3 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_second_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_second_qa[1].format(second_ethnic_group=second_ethnic_group),"audio": None})

            if index==4 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_all_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_all_qa[1].format(first_ethnic_group=first_ethnic_group, second_ethnic_group=second_ethnic_group),"audio": None})

            if index==2 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_all_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_all_only_qa[1].format(only_ethnic_group=first_ethnic_group),"audio": None})

            if index==3 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_all_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_all_only_qa[1].format(only_ethnic_group=first_ethnic_group),"audio": None})

            if index==4 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_NR_first_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_NR_first_qa[1].format(first_ethnic_group=first_ethnic_group),"audio": None})

        return newbatch
    except:
        print(traceback.format_exc(), flush=True)
        return newbatch


def mapping_MIX(batch, index):
    newbatch={"instruction":[], "answer":[]}
    try:
        for i in range(len(batch["answer"])):
            transcription=batch["answer"][i]["text"]

            if re.search("\n", transcription): num_of_speaker=2
            else: num_of_speaker=1

            first_speaker=batch["other_attributes"][i]["speaker1"]
            second_speaker=batch["other_attributes"][i]["speaker2"]

            first_gender="female" if first_speaker["gender"] in ["F", "Female"] else "male"
            second_gender="female" if second_speaker["gender"] in ["F", "Female"] else "male"

            first_ethnic_group=first_speaker["ethnic_group"]
            second_ethnic_group=second_speaker["ethnic_group"]

            if index==0:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_GR_NR_first_qa[0]), "audio": None})
                newbatch["answer"].append({"text": instructions_MIX_GR_NR_first_qa[1].format(first_ethnic_group=first_ethnic_group, first_gender=first_gender), "audio": None})

            if index==1 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_GR_NR_second_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_GR_NR_second_qa[1].format(second_ethnic_group=second_ethnic_group, second_gender=second_gender),"audio": None})

            if index==2 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_COUNT_GR_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_COUNT_GR_qa[1].format(first_gender=first_gender, second_gender=second_gender),"audio": None})

            if index==3 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_COUNT_NR_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_COUNT_NR_qa[1].format(first_ethnic_group=first_ethnic_group, second_ethnic_group=second_ethnic_group),"audio": None})

            if index==4 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_GR_NR_all_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_GR_NR_all_qa[1].format(
                    first_ethnic_group=first_ethnic_group,
                    first_gender=first_gender,
                    second_ethnic_group=second_ethnic_group,
                    second_gender=second_gender),"audio": None})

            if index==5 and num_of_speaker==2:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_COUNT_NR_GR_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_COUNT_NR_GR_qa[1].format(
                    first_ethnic_group=first_ethnic_group,
                    first_gender=first_gender,
                    second_ethnic_group=second_ethnic_group,
                    second_gender=second_gender),"audio": None})

            if index==1 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_GR_NR_all_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_GR_NR_all_only_qa[1].format(only_ethnic_group=first_ethnic_group, only_gender=first_gender),"audio": None})

            if index==2 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_COUNT_GR_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_COUNT_GR_only_qa[1].format(only_gender=first_gender),"audio": None})

            if index==3 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_COUNT_NR_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_COUNT_NR_only_qa[1].format(only_ethnic_group=first_ethnic_group),"audio": None})

            if index==4 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_COUNT_NR_GR_only_qa[0]),"audio": None})
                newbatch["answer"].append({"text": instructions_MIX_COUNT_NR_GR_only_qa[1].format(only_ethnic_group=first_ethnic_group, only_gender=first_gender),"audio": None})

            if index==5 and num_of_speaker==1:
                newbatch["instruction"].append({"text": random.choice(instructions_MIX_GR_NR_first_qa[0]), "audio": None})
                newbatch["answer"].append({"text": instructions_MIX_GR_NR_first_qa[1].format(first_ethnic_group=first_ethnic_group, first_gender=first_gender), "audio": None})

        return newbatch
    except:
        print(traceback.format_exc(), flush=True)
        return newbatch


def filter_others(batch):
    return ["Others" not in [other_attributes["speaker1"]["ethnic_group"], other_attributes["speaker2"]["ethnic_group"]] 
            for other_attributes 
            in batch["other_attributes"]]


def main(workers=100):
    
    splits = get_all_split("/mnt/home/zoux/datasets/xunlong_working_repo/swapped")
    splits.sort()
    
    print(splits, flush=True)

    for split in splits:

        if "PART6" in split:
            continue

        print("start {}".format(split), flush=True)
        ds = load_from_disk(split)
        print(len(ds), flush=True)
        ds = ds.filter(filter_others, batched=True, batch_size=100, writer_batch_size=100, num_proc=workers, desc="NR filter",)

        if not os.path.exists(split.replace("_ASR_", "_GR_")):
            ds_grs=[]
            for i in range(5):
                ds_grs.append(ds.map(mapping_GR, fn_kwargs={"index":i}, batched=True, batch_size=1, writer_batch_size=1, num_proc=workers, desc="GR mapping",))
            concatenate_datasets(ds_grs).save_to_disk(split.replace("_ASR_", "_GR_"), num_proc=10)

        if not os.path.exists(split.replace("_ASR_", "_NR_")):
            ds_nrs=[]
            for i in range(5):
                ds_nrs.append(ds.map(mapping_NR, fn_kwargs={"index":i}, batched=True, batch_size=1, writer_batch_size=1,num_proc=workers, desc="NR mapping",))
            concatenate_datasets(ds_nrs).save_to_disk(split.replace("_ASR_", "_NR_"), num_proc=10)

        if not os.path.exists(split.replace("_ASR_", "_MIX_")):
            ds_mixs=[]
            for i in range(6):
                ds_mixs.append(ds.map(mapping_MIX, fn_kwargs={"index":i}, batched=True, batch_size=1, writer_batch_size=1,num_proc=workers, desc="MIX mapping",))
            concatenate_datasets(ds_mixs).save_to_disk(split.replace("_ASR_", "_MIX_"), num_proc=10)

        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    Fire(main)
