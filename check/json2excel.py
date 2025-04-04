import json
import pandas as pd
import fire


def json2excel(json_file):
    ds_stats = json.load(open(json_file))
    dfList=[]
    for key, value in ds_stats.items():
        datasets_multimodal, split, task, dataset_name= key.split("/")[-4:]
        num_of_samples= value['num_of_samples']
        total_audio_hours= value['total_audio_hours']
        max_audio_seconds= value['max_audio_seconds']
        min_audio_seconds= value['min_audio_seconds']
        path= f"./{datasets_multimodal}/{split}/{task}/{dataset_name}"

        dfList.append([split, task, dataset_name, total_audio_hours, max_audio_seconds, min_audio_seconds, num_of_samples, path])
    df_new =  pd.DataFrame(dfList)
    df_new.to_excel(json_file.replace("ds_stats.json", "ds_stats.xlsx"), index=False, header=False)


if __name__ == "__main__":
    fire.Fire(json2excel)

