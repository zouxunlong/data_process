
import os, json
from datasets import load_from_disk
import shutil
import pandas as pd


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



def map_fn(batch):
    return {"audio_length": [len(context["audio"]["array"])/16000 for context in batch["context"]]}


file="/data/projects/13003558/zoux/workspaces/multimodal_trainer/config/dataset/meralion2_release_no_si.yaml"
ds_paths=[]
with open(file,"r") as f:
    for line in f:
        if "/train/" in line:
            data_path=line.split("path: ")[-1].strip().replace("mosaic", "hf")
            ds_paths.append(f"/data/projects/13003558/zoux/datasets/{data_path}")

print(len(ds_paths), flush=True)

ds_path_exists=[path for path in ds_paths if os.path.exists(path)]
ds_path_not_exists=[path for path in ds_paths if not os.path.exists(path)]

print(ds_path_not_exists, flush=True)

stats = {}

for ds_path in sorted(ds_path_exists):

    if os.path.exists(os.path.join(ds_path, 'ds_stats.json')):
        print(f"Reading {ds_path}", flush=True)
        curr_res = json.load(open(os.path.join(ds_path, 'ds_stats.json')))

    else:
        print('Checking {}'.format(ds_path), flush=True)
        try:
            audio_lengths = load_from_disk(ds_path)["audio_length"]
        except:
            ds=load_from_disk(ds_path)
            audio_length = ds.map(map_fn,
                                batched           = True,
                                batch_size        = 1,
                                writer_batch_size = 1,
                                num_proc          = 112,
                                desc              = f"add audio_length {os.path.basename(ds_path)}")
            ds = ds.add_column("audio_length", audio_length["audio_length"])
            
            ds.save_to_disk(ds_path+"_with_length", num_proc=10)
                        
            shutil.rmtree(ds_path, ignore_errors=False)
            
            os.rename(ds_path+"_with_length", ds_path)
            
            audio_lengths = load_from_disk(ds_path)["audio_length"]

        num_of_samples    = len(audio_lengths)
        total_audio_hours = sum(audio_lengths)/3600
        max_audio_seconds = max(audio_lengths)
        min_audio_seconds = min(audio_lengths)

        curr_res = {
            "num_of_samples"   : num_of_samples,
            "total_audio_hours": total_audio_hours,
            "max_audio_seconds": max_audio_seconds,
            "min_audio_seconds": min_audio_seconds,
        }

    split, task, dataset_name = ds_path.split('/')[-3:]

    curr_res["split"]        = split
    curr_res["task"]         = task
    curr_res["dataset_name"] = dataset_name

    with open(os.path.join(ds_path, 'ds_stats.json'), 'w') as f:
        json.dump(curr_res, f, ensure_ascii=False, indent=1)

    stats[ds_path] = curr_res

with open(os.path.join("/data/projects/13003558/zoux/ds_stats.json"), 'w') as f:
    json.dump(stats, f, ensure_ascii=False, indent=1)

json2excel("/data/projects/13003558/zoux/ds_stats.json")
print('complete all', flush=True)


