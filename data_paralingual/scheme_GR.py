
from datasets import load_from_disk, Audio, Features, Value
import random
import os
from fire import Fire
from tqdm import tqdm

questions_GR = [
    'What is the gender of the speaker?',
    "Can you tell me the speaker's gender?",
    "What gender does the speaker identify with?",
    "Could you specify the gender of the individual speaking?",
    "I'm curious about the speaker's gender, could you inform me?",
    "Would you mind sharing the gender of the speaker?",
    "What is the gender identity of the person speaking?",
    "Could you clarify the speaker's gender?",
    "I'd like to know the gender of the person who is speaking.",
    "What gender is assigned to the speaker?",
    "Could you disclose the gender of the speaker?",
    "What's the gender identity of the speaker?",
    "Do you know the gender of the person speaking?",
    "What gender pronouns does the speaker use?",
    "Can the speaker's gender be shared?",
    "Is the gender of the speaker known?",
    "May I know the gender of the speaker?",
    "What is the declared gender of the speaker?",
    "How does the speaker identify, gender-wise?",
    "What gender category does the speaker fall under?",
    "Could the gender of the speaker be mentioned?",
    "I'm interested in knowing the speaker's gender, can you share?",
    "Is it possible to state the speaker's gender?",
    "What is the speaker's gender, if you might share?",
    "Can information on the speaker's gender be provided?",
    "Would revealing the speaker's gender be possible?",
    "I wonder what the speaker's gender is, can you tell?",
    "Is there a way to know the speaker's gender?",
    "Could I inquire about the gender of the speaker?",
    "Would it be appropriate to ask for the speaker's gender?",
    "Can the gender identity of the speaker be disclosed?",
    "Is the speaker's gender something that can be shared?",
    "Might you disclose the gender of the person speaking?",
    "I'd appreciate knowing the gender of the speaker, if possible.",
    "Is revealing the speaker's gender an option?",
    "Could you let me know the gender of the speaker?",
    "What gender does the speaker prefer to be identified with?",
    "Is there information available on the speaker's gender?",
    "Could the speaker's gender identity be clarified?",
    "I'm keen to learn about the speaker's gender, may I?",
    "Are you able to provide the speaker's gender?",
    "Might the gender of the speaker be revealed?",
    "I'm looking to find out the speaker's gender, could you assist?",
    "Is it permissible to ask about the speaker's gender?",
    "Could the specifics of the speaker's gender be shared?",
    "I'm intrigued by the speaker's gender, could you enlighten me?",
    "Would you be willing to share the speaker's gender with me?",
    "Can clarification on the speaker's gender be given?",
    "Is the speaker's gender something you can divulge?",
    "Might I ask what gender the speaker identifies as?",
    "Can you provide details on the speaker's gender?",
    "Is it okay to inquire about the speaker's gender?",
    "Could you inform me about the speaker's gender?",
    "What gender identity does the speaker have?",
    "Are details on the speaker's gender available?",
    "Could you offer information about the speaker's gender?",
    "I'm curious, what is the speaker's gender?",
    "Would information regarding the speaker's gender be available?",
    "Can you enlighten me about the speaker's gender?",
    "I'd be interested in knowing the gender of the speaker, is that possible?",
    "Could the gender of the speaker be communicated?",
]


questions_NR = [
    'What is the nationality of the speaker?',
    "Can you tell me the speaker's nationality?",
    "What nationality does the speaker identify with?",
    "Could you specify the nationality of the individual speaking?",
    "I'm curious about the speaker's nationality, could you inform me?",
    "Would you mind sharing the nationality of the speaker?",
    "What is the nationality identity of the person speaking?",
    "Could you clarify the speaker's nationality?",
    "I'd like to know the nationality of the person who is speaking.",
    "What nationality is assigned to the speaker?",
    "Could you disclose the nationality of the speaker?",
    "What's the nationality identity of the speaker?",
    "Do you know the nationality of the person speaking?",
    "What nationality pronouns does the speaker use?",
    "Can the speaker's nationality be shared?",
    "Is the nationality of the speaker known?",
    "May I know the nationality of the speaker?",
    "What is the declared nationality of the speaker?",
    "How does the speaker identify, nationality-wise?",
    "What nationality category does the speaker fall under?",
    "Could the nationality of the speaker be mentioned?",
    "I'm interested in knowing the speaker's nationality, can you share?",
    "Is it possible to state the speaker's nationality?",
    "What is the speaker's nationality, if you might share?",
    "Can information on the speaker's nationality be provided?",
    "Would revealing the speaker's nationality be possible?",
    "I wonder what the speaker's nationality is, can you tell?",
    "Is there a way to know the speaker's nationality?",
    "Could I inquire about the nationality of the speaker?",
    "Would it be appropriate to ask for the speaker's nationality?",
    "Can the nationality identity of the speaker be disclosed?",
    "Is the speaker's nationality something that can be shared?",
    "Might you disclose the nationality of the person speaking?",
    "I'd appreciate knowing the nationality of the speaker, if possible.",
    "Is revealing the speaker's nationality an option?",
    "Could you let me know the nationality of the speaker?",
    "What nationality does the speaker prefer to be identified with?",
    "Is there information available on the speaker's nationality?",
    "Could the speaker's nationality identity be clarified?",
    "I'm keen to learn about the speaker's nationality, may I?",
    "Are you able to provide the speaker's nationality?",
    "Might the nationality of the speaker be revealed?",
    "I'm looking to find out the speaker's nationality, could you assist?",
    "Is it permissible to ask about the speaker's nationality?",
    "Could the specifics of the speaker's nationality be shared?",
    "I'm intrigued by the speaker's nationality, could you enlighten me?",
    "Would you be willing to share the speaker's nationality with me?",
    "Can clarification on the speaker's nationality be given?",
    "Is the speaker's nationality something you can divulge?",
    "Might I ask what nationality the speaker identifies as?",
    "Can you provide details on the speaker's nationality?",
    "Is it okay to inquire about the speaker's nationality?",
    "Could you inform me about the speaker's nationality?",
    "What nationality identity does the speaker have?",
    "Are details on the speaker's nationality available?",
    "Could you offer information about the speaker's nationality?",
    "I'm curious, what is the speaker's nationality?",
    "Would information regarding the speaker's nationality be available?",
    "Can you enlighten me about the speaker's nationality?",
    "I'd be interested in knowing the nationality of the speaker, is that possible?",
    "Could the nationality of the speaker be communicated?",
]


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def mapping(example):

    if example['Gender'] == 'm':
        answer_text = 'The speaker is a male.'
    if example['Gender'] == 'f':
        answer_text = 'The speaker is a female.'

    return {
        "context": {
            "text": None,
            "audio": example["audio"]
        },
        "instruction": {
            "text": random.choice(questions_GR),
            "audio": None
        },
        "answer": {
            "text": answer_text,
            "audio": None
        },
        "other_attributes": {
            "Gender": example["Gender"],
            "Nationality": example["Nationality"],
            "VGGFace1 ID": example["VGGFace1 ID"],
            "VoxCeleb1 ID": example["VoxCeleb1 ID"],
            "index": example["index"],
        }
    }


def map2schema(split, workers=32):

    print(f"start {split}", flush=True)

    ds = load_from_disk(split)

    problem_ids = []
    for _i in tqdm(range(len(ds)), desc="checking samples"):
        try:
            sample = ds[_i]
            if not sample["Gender"] in ['m', 'f']:
                raise ValueError
        except:
            problem_ids.append(_i)
    ds = ds.select([_i for _i in range(len(ds)) if _i not in problem_ids])

    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "Gender": ds.features["Gender"],
            "Nationality": ds.features["Nationality"],
            "VGGFace1 ID": ds.features["VGGFace1 ID"],
            "VoxCeleb1 ID": ds.features["VoxCeleb1 ID"],
            "index": ds.features["index"],
        }
    })

    ds = ds.map(mapping,
                features=features,
                remove_columns=ds.column_names,
                num_proc=workers,
                batch_size=1,
                writer_batch_size=1,
                keep_in_memory=False,
                load_from_cache_file=True
                )

    ds.save_to_disk(split.replace("VoxCeleb1", "VoxCeleb1_GR_v1"), num_proc=4)
    print(f"complete {split}", flush=True)


def main(dir):
    splits = get_all_split(dir)
    splits.sort()
    for split in splits:
        map2schema(split)


if __name__ == "__main__":
    Fire(main)
