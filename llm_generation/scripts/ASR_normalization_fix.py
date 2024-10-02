
from pprint import pprint
import random
import re
import fire
from datasets import load_from_disk, Value, concatenate_datasets
from openai import OpenAI
from glob import glob
import os

class Reg_Exp:
    pattern_punctuation = r"""[!?,*.:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""
    pattern_url = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    pattern_email = r"[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}"
    pattern_arabic = r"[\u0600-\u06FF]"
    pattern_chinese = r"[\u4e00-\u9fff]"
    pattern_tamil = r"[\u0B80-\u0BFF]"
    pattern_thai = r"[\u0E00-\u0E7F]"
    pattern_russian = r"[\u0400-\u04FF]"
    pattern_korean = r"[\uac00-\ud7a3]"
    pattern_japanese = r"[\u3040-\u30ff\u31f0-\u31ff]"
    pattern_vietnamese = r"[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]"
    pattern_emoji = r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F64F\U0001F680-\U0001FAFF\U00002702-\U000027B0]'

def map_fn(sample):

    normalized = sample['answer']['text']
    origin = sample['other_attributes']['transcription']
    
    trimed_origin = ' '.join(re.sub(r"{}|{}|{}".format(
        Reg_Exp.pattern_url,
        Reg_Exp.pattern_email,
        Reg_Exp.pattern_punctuation
    ), " ", origin, 0, re.I).split()).strip().lower()

    trimed_normalized = ' '.join(re.sub(r"{}|{}|{}".format(
        Reg_Exp.pattern_url,
        Reg_Exp.pattern_email,
        Reg_Exp.pattern_punctuation
    ), " ", normalized, 0, re.I).split()).strip().lower()



    if trimed_origin==trimed_normalized:
        sample['answer']['text'] = normalized
    else:
        print(origin, flush=True)
        print(normalized, flush=True)
        print(trimed_origin, flush=True)
        print(trimed_normalized, flush=True)
        breakpoint()

    return sample

def filter_fn(example):
    return example['answer']['text'].strip() not in ['Template not matched.', '']

def fix(split, num_proc=1):

    ds = load_from_disk(split)

    features = ds.features

    ds = ds.map(
        map_fn,
        features          = features,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc="fix for {}".format(split),
    )

    ds = ds.filter(
        filter_fn,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "filter empty answers",
    )

    # ds.save_to_disk(split.replace("ASR_normalized_opus", "ASR_normalized_fixed"), num_proc=4)


def main(pattern="/mnt/data/all_datasets/ASR_normalized_opus/test/ASR/librispeech_clean_ASR_v2"):
    splits = glob(pattern)
    splits.sort()
    pprint(splits)
    for split in splits:
        if os.path.exists(split.replace("ASR_normalized_opus", "ASR_normalized_fixed")):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        fix(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
