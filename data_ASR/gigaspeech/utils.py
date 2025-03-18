from datasets import load_from_disk
from pprint import pprint
from glob import glob
import os, random

asr_instructions_vi = ['Can you write out the speech in Vietnamese?',
                       'Capture the spoken content in writing in Vietnamese.',
                       'Convert the speech into a written transcript in Vietnamese.',
                       'Could you put the speech into writing in Vietnamese?',
                       'Please convert the audio into written words in Vietnamese.',
                       'Record the speech in written form in Vietnamese.',
                       'Transcribe the spoken content into text in Vietnamese.',
                       'Write down what was said in Vietnamese, please.']

asr_instructions_id = ['Can you write out the speech in Indonesian?',
                       'Capture the spoken content in writing in Indonesian.',
                       'Convert the speech into a written transcript in Indonesian.',
                       'Could you put the speech into writing in Indonesian?',
                       'Please convert the audio into written words in Indonesian.',
                       'Record the speech in written form in Indonesian.',
                       'Transcribe the spoken content into text in Indonesian.',
                       'Write down what was said in Indonesian, please.']

asr_instructions_th = ['Can you write out the speech in Thai?',
                       'Capture the spoken content in writing in Thai.',
                       'Convert the speech into a written transcript in Thai.',
                       'Could you put the speech into writing in Thai?',
                       'Please convert the audio into written words in Thai.',
                       'Record the speech in written form in Thai.',
                       'Transcribe the spoken content into text in Thai.',
                       'Write down what was said in Thai, please.']

asr_instructions_ta = ['Can you write out the speech in Tamil?',
                       'Capture the spoken content in writing in Tamil.',
                       'Convert the speech into a written transcript in Tamil.',
                       'Could you put the speech into writing in Tamil?',
                       'Please convert the audio into written words in Tamil.',
                       'Record the speech in written form in Tamil.',
                       'Transcribe the spoken content into text in Tamil.',
                       'Write down what was said in Tamil, please.']

asr_instructions_ms = ['Can you write out the speech in Malay?',
                       'Capture the spoken content in writing in Malay.',
                       'Convert the speech into a written transcript in Malay.',
                       'Could you put the speech into writing in Malay?',
                       'Please convert the audio into written words in Malay.',
                       'Record the speech in written form in Malay.',
                       'Transcribe the spoken content into text in Malay.',
                       'Write down what was said in Malay, please.']

asr_instructions_zh = ['Can you write out the speech in Chinese?',
                       'Capture the spoken content in writing in Chinese.',
                       'Convert the speech into a written transcript in Chinese.',
                       'Could you put the speech into writing in Chinese?',
                       'Please convert the audio into written words in Chinese.',
                       'Record the speech in written form in Chinese.',
                       'Transcribe the spoken content into text in Chinese.',
                       'Write down what was said in Chinese, please.']


def map_id(instructions):
    new_instructions=[{"text":random.choice(asr_instructions_id), "audio":None} for _ in instructions]
    return {"instruction": new_instructions}

def map_th(instructions):
    new_instructions=[{"text":random.choice(asr_instructions_th), "audio":None} for _ in instructions]
    return {"instruction": new_instructions}

def map_vi(instructions):
    new_instructions=[{"text":random.choice(asr_instructions_vi), "audio":None} for _ in instructions]
    return {"instruction": new_instructions}

dss=glob("/mnt/data/all_datasets/data_process/_data_in_processing/gigaspeech2/id/test")
for ds_path in sorted(dss):


    ds = load_from_disk(ds_path)
    if ds[0]["instruction"]["text"] != "":
        continue
    if ds_path.split("/")[-2]=="id":
        ds=ds.map(map_id, 
                  input_columns=["instruction"], 
                  remove_columns=["instructions"],
                  batched=True, 
                  writer_batch_size=1, 
                  num_proc=16)

    if ds_path.split("/")[-2]=="vi":
        ds=ds.map(map_vi, 
                  input_columns=["instruction"], 
                  remove_columns=["instructions"],
                  batched=True, 
                  writer_batch_size=1, 
                  num_proc=16)

    if ds_path.split("/")[-2]=="th":
        ds=ds.map(map_th, 
                  input_columns=["instruction"], 
                  remove_columns=["instructions"],
                  batched=True, 
                  writer_batch_size=1, 
                  num_proc=16)

    ds.save_to_disk(ds_path+"_new", num_proc=4)


    print(f"Done {ds_path}")


