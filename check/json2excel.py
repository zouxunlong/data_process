import json
import pandas as pd


def json2excel(json_file):
    ds_stats = json.load(open(json_file))
    dfList=[]
    for key, value in ds_stats.items():
        datasets_multimodal, other_prepared, ASQA, WavCaps_ASQA_v2= key.split("/")[:5]
        if other_prepared=="other_prepared":
            split= ".".join(key.split("/")[8:])
        else:
            split= other_prepared
        num_of_samples= value['num_of_samples']
        total_audio_hours= value['total_audio_hours']
        max_audio_seconds= value['max_audio_seconds']
        min_audio_seconds= value['min_audio_seconds']
        path= f"/{datasets_multimodal}/{other_prepared}/{ASQA}/{WavCaps_ASQA_v2}"

        dfList.append([other_prepared, ASQA, WavCaps_ASQA_v2, split, total_audio_hours, max_audio_seconds, min_audio_seconds, num_of_samples, path])
    df_new =  pd.DataFrame(dfList)
    df_new.to_excel('nlb_stats.xlsx', index=False, header=False)


if __name__ == "__main__":

    json2excel("/mnt/data/all_datasets/datasets_hf_array/datasets_multimodal/ds_stats.json")

    