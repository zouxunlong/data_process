import io
import os
import re
from datasets import load_from_disk, Audio, Value, Features

mappings_er = {
    "happy"       : "happy",
    "joy"         : "happy",
    "happiness"   : "happy",
    "sad"         : "sad",
    "neutral"     : "neutral",
    "fearful"     : "fearful",
    "fear"        : "fearful",
    "embarrassed" : "embarrassed",
    "scared"      : "scared",
    "excited"     : "excited",
    "confusion"   : "confusion",
    "frustrated"  : "frustrated",
    "frustration" : "frustrated",
    "disgusted"   : "disgusted",
    "disgust"     : "disgusted",
    "disappointed": "disappointed",
    "angry"       : "angry",
    "anger"       : "angry",
    "surprised"   : "surprised",
    "surprise"    : "surprised"
}

mappings_gr = {
    "male"                           : "male",
    "m"                              : "male",
    "0"                              : "male",
    "male_masculine"                 : "male",
    "female"                         : "female",
    "female_feminine"                : "female",
    "do_not_wish_to_say"             : "female",
    "f"                              : "female",
    "1"                              : "female",
    "yes, there are female speakers.": "female",
}


# def filter_fn(answers, other_attributess):
#     return [bool(str(other_attributes["gender"]).strip().lower().split(",")[0] in mappings_gr) for other_attributes in other_attributess]



def filter_fn(answers, other_attributess):
    return [re.search(r"(male|female)", a["text"].lower()) for a in answers]


# def filter_fn(answers, other_attributess):
#     return [bool(str(other_attributes["Emotion"]).strip().lower().split(",")[0] in mappings_er) for other_attributes in other_attributess]


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(other_attributes):
#         match = mappings_gr[str(a["gender"]).strip().lower().split(",")[0]]
#         other_attributes[i]["gender"] = match
#     return {"other_attributes": other_attributes}


def map_fn(answers, other_attributes):
    for i, a in enumerate(answers):
        # Extract emotions from the text using regex
        match = re.search(r"(male|female)", a["text"].lower())
        other_attributes[i]["gender"] = mappings_gr[match.group(1)]

    return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(other_attributes):
#         match = mappings_gr[str(a["speaker"]["gender"]).strip().lower()]
#         other_attributes[i]["gender"] = match
#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(other_attributes):
#         match = mappings_gr[a["Gender"].strip().lower()]
#         other_attributes[i]["gender"] = match
#         other_attributes[i].pop("Gender")
#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(answers):
#         match = mappings_gr[a["text"].strip().lower()]
#         other_attributes[i]["gender"] = match
#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(other_attributes):
#         match = mappings_er[a["emotion"].strip().lower()]
#         other_attributes[i]["emotion"] = match
#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(answers):
#         match = mappings_er[a["text"].strip().lower()]
#         other_attributes[i]["emotion"] = match
#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(other_attributes):
#         match = mappings_er[a["emotion_class"].strip().lower()]
#         other_attributes[i]["emotion"] = match
#     return {"other_attributes": other_attributes}


# def map_fn(answers, instruction, other_attributes):
#     for i, a in enumerate(answers):
#         # Extract emotions from the text using regex
#         match = re.search(r"(happy|joy|happiness|sad|neutral|fearful|fear|embarrassed|scared|excited|confusion|frustrated|disgusted|disgust|disappointed|neutral|angry|anger|surprised|surprise|frustration)", a["text"].lower())
#         if match:
#             other_attributes[i]["emotion"] = mappings_er[match.group(1)]
#         else:
#             print(a["text"].lower(), flush=True)
#             breakpoint()
#             other_attributes[i]["emotion"] = "neutral"
#     return {"other_attributes": other_attributes}

from glob import glob

ds_paths=glob("/data/projects/13003558/zoux/datasets/sea_audiobench/PQA/test_subset1000-short/gr_*")

for ds_path in sorted(ds_paths):
    ds_path_to_save=f"/data/projects/13003558/zoux/datasets/er_gr_test/gr/{os.path.basename(ds_path)}"
    ds_name=os.path.basename(ds_path)

    if os.path.exists(ds_path_to_save):
        print(f"{ds_name} exists", flush=True)
        continue

    ds = load_from_disk(ds_path)
    print(ds_name, flush=True)
    print(ds[0]["other_attributes"], flush=True)

    # emotion_labels=[o["emotion_class"] for o in ds["other_attributes"]]
    # print(ds, flush=True)
    # print(set(emotion_labels), flush=True)

    # features=ds.features
    # features["other_attributes"]["emotion"]=Value(dtype='string', id=None)

    # ds=ds.filter(filter_fn, batched=True, input_columns=["answer", "other_attributes"], num_proc=1)
    # ds=ds.map(map_fn, batched=True, input_columns=["answer", "instruction", "other_attributes"], num_proc=1)
    # ds.save_to_disk(ds_path_to_save, num_proc=1)

    # gender_labels=[o["emotion"] for o in ds["other_attributes"]]
    # print(ds, flush=True)
    # print(set(gender_labels), flush=True)
    # print(ds[0]["other_attributes"], flush=True)



    # gender_labels=[o["gender"] for o in ds["other_attributes"]]
    # print(ds, flush=True)
    # print(set(gender_labels), flush=True)

    ds=ds.filter(filter_fn, batched=True, input_columns=["answer", "other_attributes"], num_proc=1)
    ds=ds.map(map_fn, batched=True, input_columns=["answer", "other_attributes"], num_proc=1)
    ds.save_to_disk(ds_path_to_save, num_proc=1)

    gender_labels=[o["gender"] for o in ds["other_attributes"]]
    print(ds, flush=True)
    print(set(gender_labels), flush=True)
    print(ds[0]["other_attributes"], flush=True)








# from glob import glob
# ds_paths=glob("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/er/*")

# for ds_path in ds_paths:
#     print(f"Loading dataset from {ds_path}")
#     ds=load_from_disk(ds_path)
#     print(ds.features["context"])


# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_openslr_ta_30")
# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_vietnam_celeb_30")
# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_fleurs_km_30")
# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_fleurs_en_30")

