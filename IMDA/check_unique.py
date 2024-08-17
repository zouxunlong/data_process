from datasets import load_from_disk
import json
from collections import defaultdict
from tqdm import tqdm


def check(dataset_path):
    print("start {}".format(dataset_path), flush=True)
    res=defaultdict(set)
    ds_dict=load_from_disk(dataset_path).select_columns(["speaker1", "speaker2"])
    for split, ds in ds_dict.items():
        print(split, flush=True)
        for item in ds:
            res["gender"].add(item["speaker1"]["gender"])
            res["ethnic_groups"].add(item["speaker1"]["ethnic_group"])
            res["gender"].add(item["speaker2"]["gender"])
            res["ethnic_groups"].add(item["speaker2"]["ethnic_group"])
    res["gender"]=list(res["gender"])
    res["ethnic_groups"]=list(res["ethnic_groups"])
    print(res, flush=True)
    return res


def check2(dataset_path):
    print("start {}".format(dataset_path), flush=True)
    res=defaultdict(set)
    ds_dict=load_from_disk(dataset_path).select_columns("speaker")
    for split, ds in ds_dict.items():
        print(split, flush=True)
        for item in ds:
            res["gender"].add(item["speaker"]["gender"])
            res["ethnic_groups"].add(item["speaker"]["ethnic_group"])
    res["gender"]=list(res["gender"])
    res["ethnic_groups"]=list(res["ethnic_groups"])
    print(res, flush=True)
    return res


def check3(dataset_path):
    print("start {}".format(dataset_path), flush=True)
    res=defaultdict(set)
    ds=load_from_disk(dataset_path).select_columns("other_attributes")
    for item in tqdm(ds):
        res["conversation_id"].add(item["other_attributes"]["conversation_id"][1:])
    res["conversation_id"]=list(res["conversation_id"])
    print(res, flush=True)
    return res


if __name__ == "__main__":
    # result={}
    # for part in ["PART3","PART4","PART5"]:
    #     res=check("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/IMDA.hf/{}.hf".format(part))
    #     result[part]=res

    # for part in ["PART1","PART2"]:
    #     res=check2("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/IMDA.hf/{}.hf".format(part))
    #     result[part]=res
    # with open("res.jsonl", "w", encoding="utf8") as f:
    #     json.dump(result, f, indent=4)

    result={}
    for part in ["PART1","PART2"]:
        res=check3("/home/all_datasets/datasets_multimodal/ASR/IMDA.ASR.schemed/{}.ASR.schemed/test".format(part))
        result[part]=res
    with open("res.conversation_id.jsonl", "w", encoding="utf8") as f:
        json.dump(result, f, indent=4)