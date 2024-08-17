from tqdm import tqdm
import pandas as pd
from datasets import Dataset, Value, Audio, Features
from collections import defaultdict


features = Features({'answer': {'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
                                'text': Value(dtype='string', id=None)},
                     'context': {'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
                                 'text': Value(dtype='string', id=None)},
                     'instruction': {'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
                                     'text': Value(dtype='string', id=None)},
                     'other_attributes': {}
                     })


def csv2hf(file, split):
    csv_file = pd.read_csv(file)
    aq2as_dict = defaultdict(list)
    for i, (file_name, QuestionText, answer, confidence) in enumerate(tqdm(csv_file.values)):
        if confidence in ["yes", "Yes"]:
            aq2as_dict[("audio_files/"+file_name, QuestionText)].append(answer)
    print(i)
    print(len(aq2as_dict.items()))
    aq2a_dict = {}
    for key, answers in aq2as_dict.items():
        answer = max(set(answers), key=answers.count)
        if len(answers) == 1:
            aq2a_dict[key] = answer
        if len(answers) == 2 and answers.count(answer) == 2:
            aq2a_dict[key] = answer
        if len(answers) == 3 and answers.count(answer) >= 2:
            aq2a_dict[key] = answer
    print(len(aq2a_dict.items()))
    ds_dict = defaultdict(list)
    for (audio, question), answer in aq2a_dict.items():
        ds_dict["context"].append({"audio": audio, "text": None})
        ds_dict["instruction"].append({"audio": None, "text": question})
        ds_dict["answer"].append({"audio": None, "text": answer})
        ds_dict["other_attributes"].append({})
    ds = Dataset.from_dict(mapping=ds_dict, features=features, split=split)
    ds.save_to_disk("clotho_aqa.schemed/{}".format(split))


csv2hf("clotho_aqa_train.csv", "train")
