from datasets import Audio, Dataset, Features, Value
import fire
from tqdm import tqdm
import os, json
from glob import glob




def build_hf(
        root="/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data",
        output_path="/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_asr",
):
    transcriptions={}
    transcriptions_dict = json.load(open("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/transcripts.json"))
    for key, value in transcriptions_dict.items():
        for k, v in value.items():
            transcriptions[k]=v

    wav_files = glob(f"{root}/*.wav")

    ds_dict = {
        "context"    : ["/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/kaiser_test1.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/kaiser_test2.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/Kaiser_test3.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/Kaiser_test3crowd.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/Kaiser_test3machine.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/Kaiser_test3music.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/Kaiser_test3thunder.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/Kaiser_test3traffic.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/test4_bto.wav",
                        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/ste_data/test4_coe_car_food.wav"],
        "instruction": ["Please transcribe."]*10,
        "answer"     : [transcriptions["kaiser_test1.wav"],
                        transcriptions["kaiser_test2.wav"],
                        transcriptions["Kaiser_test3.wav"],
                        transcriptions["Kaiser_test3.wav"],
                        transcriptions["Kaiser_test3.wav"],
                        transcriptions["Kaiser_test3.wav"],
                        transcriptions["Kaiser_test3.wav"],
                        transcriptions["Kaiser_test3.wav"],
                        transcriptions["test4_bto.wav"],
                        transcriptions["test4_coe_car_food.wav"]
                        ],
        "testset_name" : ["kaiser_test1",
                        "kaiser_test2",
                        "Kaiser_test3",
                        "Kaiser_test3crowd",
                        "Kaiser_test3machine",
                        "Kaiser_test3music",
                        "Kaiser_test3thunder",
                        "Kaiser_test3traffic",
                        "test4_bto",
                        "test4_coe_car_food"
                        ],
    }

    features = Features({
        'context'     : Audio(sampling_rate=16000, decode=True),
        'instruction' : Value(dtype='string'),
        'answer'      : Value(dtype='string'),
        'testset_name': Value(dtype='string')
    })

    ds = Dataset.from_dict(ds_dict, features=features)
    print(ds, flush=True)
    ds.save_to_disk(output_path)
    print("Saved to {}".format(output_path), flush=True)


if __name__ == "__main__":
    fire.Fire(build_hf)
