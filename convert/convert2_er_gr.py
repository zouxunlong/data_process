import io
import os
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


def filter_fn(answers, other_attributess):

    return [bool(str(other_attributes["gender"]).strip().lower().split(",")[0] in mappings_gr) for other_attributes in other_attributess]


def map_fn(answers, other_attributes):

    for i, a in enumerate(other_attributes):
        match = mappings_gr[str(a["gender"]).strip().lower().split(",")[0]]
        other_attributes[i]["gender"] = match
        # other_attributes[i].pop("GENDER")
    return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     breakpoint()
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
#     for i, a in enumerate(answers):
#         match = mappings[a["text"].strip().lower()]
#         other_attributes[i]["emotion"] = match
#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(other_attributes):
#         match = mappings[a["emotion_class"].strip().lower()]
#         other_attributes[i]["emotion"] = match

#     return {"other_attributes": other_attributes}


# def map_fn(answers, other_attributes):
#     for i, a in enumerate(answers):
#         # Extract emotions from the text using regex
#         match = re.search(r"(happy|joy|happiness|sad|neutral|fearful|fear|embarrassed|scared|excited|confusion|frustrated|disgusted|disgust|disappointed|neutral|angry|anger|surprised|surprise|frustration)", a["text"])
#         if match:
#             other_attributes[i]["emotion"] = mappings[match.group(1)]
#         else:
#             other_attributes[i]["emotion"] = "neutral"
#     return {"other_attributes": other_attributes}


# ds_path="/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/datasets_multimodal/test/ASR/fleurs_malay_ms_30_ASR"
# ds_path_to_save=f"/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/{os.path.basename(ds_path)}"

# ds=load_from_disk(ds_path)

# features=ds.features
# features["other_attributes"]["gender"]=Value(dtype='string', id=None)

# ds=ds.filter(filter_fn, batched=True, input_columns=["answer", "other_attributes"], num_proc=1)
# ds=ds.map(map_fn, batched=True, features=features, input_columns=["answer", "other_attributes"], num_proc=1)
# ds.save_to_disk(ds_path_to_save, num_proc=1)

# print(ds[0])

from glob import glob
ds_paths=glob("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/er/*")

for ds_path in ds_paths:
    print(f"Loading dataset from {ds_path}")
    ds=load_from_disk(ds_path)
    print(ds.features["context"])


# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_openslr_ta_30")
# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_vietnam_celeb_30")
# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_fleurs_km_30")
# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2.1/er_gr_test/gr/gr_fleurs_en_30")

