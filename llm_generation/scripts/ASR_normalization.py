
from pprint import pprint
import random
import fire
from datasets import load_from_disk, Value, concatenate_datasets
from openai import OpenAI
from glob import glob
import os


def map_fn(sample):

    port = random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    
    # Question generation

    text = sample['answer']['text']

    TEMPLATE = """\
        [Transcription]
        {context}

        [Task]
        You task is to normalize the transcription to standard writing format.
        Pay attention to proper capitalizations and punctuations.
        Do not change the original code-switch text and speaker tags if any.
        Do not change any word or charactor.
        Do not output anything other than the normalized Transcription.
        """

    prompt_sample = TEMPLATE.format(context=text)

    chat_response = client.chat.completions.create(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[{"role": "user", "content": prompt_sample},]
        )

    transcription=chat_response.choices[0].message.content
        
    return {
        'answer': {'text': transcription, 'audio': None},
        'other_attributes': {'transcription': sample['answer']['text']}
    }

def filter_fn(example):
    return example['answer']['text'].strip() not in ['Template not matched.', '']

def generation(split, num_proc=128):

    ds = load_from_disk(split)

    features = ds.features
    features['other_attributes'] = {"transcription": Value(dtype='string')}

    ds = ds.map(
        map_fn,
        features          = features,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc="ASR normalization for {}".format(split),
    )

    ds = ds.filter(
        filter_fn,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "filter empty answers",
    )

    ds.save_to_disk(split.replace("datasets_multimodal", "datasets_multimodal_norm"), num_proc=4)


def main(pattern):
    splits = glob(pattern)
    splits.sort()
    pprint(splits)
    for split in splits:
        if os.path.exists(split.replace("datasets_multimodal", "datasets_multimodal_norm")):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        generation(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
