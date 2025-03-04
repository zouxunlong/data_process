from datasets import load_dataset, Dataset, Audio,Features, Value, ClassLabel
import json
import random
from pathlib import Path
iemocap = load_dataset("Ar4ikov/iemocap_audio_text")

with open('/home/zhuohan/2024LLM/processed-datasets/iemocap_v1/emo_instructions.json', 'r') as f:
    emo_instruction = json.load(f)
with open('/home/zhuohan/2024LLM/processed-datasets/iemocap_v1/gen_instructions.json', 'r') as f:
    gen_instruction = json.load(f)

## extend labels' name
expend_labels = {
        'sad': 'sad',
        'fru': 'frustration',
        'neu': 'neutral state',
        'hap': 'happiness',
        'exc': 'excited',
        'sur': 'surprise',
        'ang': 'anger', 
        'fea': 'fear',
        'dis': 'disgust',
        'oth': 'other'
    }

def map_function(example):
    
    # 1. get gender information
    gender_code = example['titre'].split('_')[-1][0]
    if gender_code == 'M':
        gender = "male"
    elif gender_code == "F":
        gender = "female"
    
    gen_sample = random.sample(range(len(gen_instruction)),1)
    gen_selection = gen_instruction[str(gen_sample[0])]
    
    example['instruction_gen'] = {
        'audio': None,
        'text': gen_selection['Question']
    }

    example['answer_gen'] = {
        'audio': None,
        'text': gen_selection['Answer'].replace("#XXXX#", gender.lower())
    }

    #2. get emotion information
    emo_sample = random.sample(range(len(emo_instruction)),1)
    emo_selection = emo_instruction[str(emo_sample[0])]
    
    example['instruction_emo'] = {
        'audio': None,
        'text': emo_selection['Question']
    }

    example['answer_emo'] = {
        'audio': None,
        'text': emo_selection['Answer'].replace("#XXXX#", expend_labels[example['emotion']].lower())
    }
    
    # 3. reformat other attributes
    example['context'] = {
            'text': example['to_translate'],
            'audio': example['audio']
        }
    example['other_attributes'] = {
        'Audio ID': example['audio']['path'],
        'Emotion': expend_labels[example['emotion']],
        'Gender': gender,
        'Start Time': example['start_time'],
        'End Time': example['end_time'],
        'Translated': example['translated']

    }

    # breakpoint()

    return example


# update feature types
updated_features=Features({
    'context': {'text': Value(dtype='string', id=None),
                'audio': Audio(sampling_rate=16000)},
    'instruction_gen': {'audio':  Audio(sampling_rate=16000),
                        'text': Value(dtype='string', id=None)},
    'answer_gen': {'audio':  Audio(sampling_rate=16000),
                    'text': Value(dtype='string', id=None)},
    'instruction_emo': {'audio':  Audio(sampling_rate=16000),
                        'text': Value(dtype='string', id=None)},
    'answer_emo': {'audio':  Audio(sampling_rate=16000),
                    'text': Value(dtype='string', id=None)},
    'other_attributes': {'Audio ID': Value(dtype='string', id=None),
                            'Emotion': Value(dtype='string', id=None),
                            # ClassLabel(num_classes=10, 
                            #                         names=['sad', 'frustration', 'neutral state', 'happiness', 'excited', 'surprise', 
                            #                                 'anger', 'fear', 'disgust', 'other'], 
                            #                         names_file=None, id=None),
                            'Gender': Value(dtype='string', id=None),
                            # ClassLabel(num_classes=2,
                            #                         names=['female', 'male'],
                            #                         names_file=None, id=None),
                            'Start Time': Value(dtype='float64', id=None),
                            'End Time': Value(dtype='float64', id=None),
                            'Translated': Value(dtype='string', id=None),
                        }
})

iemocap = iemocap['train']

# breakpoint()
remove_columns = iemocap.column_names
iemocap = iemocap.map(map_function,
                      remove_columns = remove_columns,
                      features = updated_features,
                      num_proc = 32,
                      batch_size = 1,
                      writer_batch_size = 1,)

# save whole
output_whole = "/home/zhuohan/2024LLM/processed-datasets/iemocap_1119/whole"
Path(output_whole).mkdir(parents=True, exist_ok=True)
iemocap.save_to_disk(output_whole)

# split train test
ds = iemocap.train_test_split(test_size=0.1, shuffle=True)
iemocap_train = ds['train']
iemocap_test = ds['test']

# breakpoint()

iemocap_emo_train = iemocap_train.select_columns(["context", "instruction_emo", "answer_emo", "other_attributes"])
iemocap_emo_test = iemocap_test.select_columns(["context", "instruction_emo", "answer_emo", "other_attributes"])

iemocap_emo_train = iemocap_emo_train.rename_column("instruction_emo", "instruction")
iemocap_emo_train = iemocap_emo_train.rename_column("answer_emo", "answer")
iemocap_emo_test = iemocap_emo_test.rename_column("instruction_emo", "instruction")
iemocap_emo_test = iemocap_emo_test.rename_column("answer_emo", "answer")


iemocap_gen_train = iemocap_train.select_columns(["context", "instruction_gen", "answer_gen", "other_attributes"])
iemocap_gen_test = iemocap_test.select_columns(["context", "instruction_gen", "answer_gen", "other_attributes"])

iemocap_gen_train = iemocap_gen_train.rename_column("instruction_gen", "instruction")
iemocap_gen_train = iemocap_gen_train.rename_column("answer_gen", "answer")
iemocap_gen_test = iemocap_gen_test.rename_column("instruction_gen", "instruction")
iemocap_gen_test = iemocap_gen_test.rename_column("answer_gen", "answer")


# save emo
output_emo_train = "/home/zhuohan/2024LLM/processed-datasets/iemocap_1119/emo/train"
output_emo_test = "/home/zhuohan/2024LLM/processed-datasets/iemocap_1119/emo/test"
Path(output_emo_train).mkdir(parents=True, exist_ok=True)
Path(output_emo_test).mkdir(parents=True, exist_ok=True)

iemocap_emo_train.save_to_disk(output_emo_train)
iemocap_emo_test.save_to_disk(output_emo_test)

# save gen
output_gen_train = "/home/zhuohan/2024LLM/processed-datasets/iemocap_1119/gen/train"
output_gen_test = "/home/zhuohan/2024LLM/processed-datasets/iemocap_1119/gen/test"
Path(output_gen_train).mkdir(parents=True, exist_ok=True)
Path(output_gen_test).mkdir(parents=True, exist_ok=True)

iemocap_gen_train.save_to_disk(output_gen_train)
iemocap_gen_test.save_to_disk(output_gen_test)

breakpoint()