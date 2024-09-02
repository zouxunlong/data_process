import random
from datasets import load_from_disk
from pprint import pprint

from fire import Fire

AC_instructions = ['What specific sounds can be heard in this audio recording?',
                   'List the specific noises that can be detected in this recording.',
                   'What is the main message of the audio?',
                   'Describe what is being heard in this audio.',
                   'Describe the environmental sounds in the audio.',
                   'Please transcribe the audio.',
                   'Please provide a written version of this audio.',
                   'Could you convert this audio into text?',
                   'What is the central theme of this audio?',
                   'Explain the sounds heard in this audio.',
                   'Identify the distinct sounds present in this audio clip.',
                   'Can you describe the events happening in the audio?',
                   'What is occurring in the sound recording?',
                   'Which sounds are identifiable in this sound clip?',
                   'What key information does the audio communicate?',
                   'Could you transcribe the contents of this audio?',
                   'What is the primary message conveyed in this audio?',
                   'Detail the ambient sounds included in the audio.',
                   'Outline the environmental acoustics captured in the audio.',
                   'Please tell me what is going on in the audio.']


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def mapping(example):

    return {
        "instruction": {
            "text": random.choice(AC_instructions),
            "audio": None
        },
        "answer": {
            "text": example["other_attributes"]["caption"],
            "audio": None
        }
    }


def map2schema(split, workers=32):

    print(f"start {split}", flush=True)

    ds = load_from_disk(split)
    print(ds, flush=True)

    ds = ds.map(mapping,
                features=ds.features,
                num_proc=workers,
                batch_size=1,
                writer_batch_size=1,
                keep_in_memory=False,
                load_from_cache_file=True
                )
    print(ds, flush=True)
    ds.save_to_disk(split.replace("_ASQA_", "_AC_"), num_proc=4)
    print(f"complete {split}", flush=True)


def main():

    map2schema("/mnt/data/all_datasets/datasets_multimodal/train/ASQA/AudioCaps_ASQA_v2")
    map2schema("/mnt/data/all_datasets/datasets_multimodal/train/ASQA/WavCaps_ASQA_v2")

if __name__ == "__main__":
    Fire(main)
