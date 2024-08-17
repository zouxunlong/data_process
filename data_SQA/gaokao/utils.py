from datasets import Audio, Value, Dataset, load_from_disk
from tqdm import tqdm
import pandas as pd
from pprint import pprint


features=load_from_disk("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/gaokao/cn_college_english_exam.schemed").features

ds_dict = {
    "context": [],
    "instruction": [],
    "answer": [],
    "other_attributes": [],
}

with open("gaokao.tsv") as file:
    tsv_file = pd.read_csv(file, delimiter="\t")
    for id, audio, n_frames, instruction, choice_A, choice_B, choice_C, choice_answer, with_speech in tqdm(tsv_file.values):
        choice_dict={"A": choice_A, "B": choice_B, "C": choice_C}
        ds_dict["context"].append({"audio": audio, "text": None})
        ds_dict["instruction"].append({"audio": None, "text": instruction})
        ds_dict["answer"].append({"audio": None, "text": choice_dict[choice_answer][4:]})
        ds_dict["other_attributes"].append({
            "audio_name": audio, 
            "choices": "\n".join([choice_A, choice_B, choice_C]),
            "mc_answer": choice_dict[choice_answer]
            })
ds = Dataset.from_dict(ds_dict, features=features)
print(ds, flush=True)
ds.save_to_disk("gaokao.hf", num_proc=10)