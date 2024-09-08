
from glob import glob
from datasets import load_from_disk, Audio, Features, Value
import random
import os
from fire import Fire
from tqdm import tqdm


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

    return {
        "answer": {
            "text": f'Base on the accent, the speaker may be from {example["other_attributes"]["Nationality"]}.',
            "audio": None
        }
    }


def map2schema(split, workers=64):

    print(f"start {split}", flush=True)

    ds = load_from_disk(split)

    ds = ds.map(mapping,
                features=ds.features,
                num_proc=workers,
                batch_size=1,
                writer_batch_size=1,
                keep_in_memory=False,
                load_from_cache_file=True
                )

    ds.save_to_disk(split.replace("VoxCeleb1_NR_v2", "VoxCeleb1_NR_v1"), num_proc=4)
    print(f"complete {split}", flush=True)


def main(pattern):

    splits = glob(pattern)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("VoxCeleb1_NR_v2", "VoxCeleb1_NR_v1")):
            print("complete {}".format(split), flush=True)
            continue
        map2schema(split)

if __name__ == "__main__":
    Fire(main)
