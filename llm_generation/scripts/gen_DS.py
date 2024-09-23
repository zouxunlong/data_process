
from glob import glob
import os
import fire
from datasets import load_from_disk, Value
import random
from openai import OpenAI
import random


candidate_instructions = [
    "Create a brief summary capturing the key points and decisions made during the dialogue.",
    "Condense the dialogue into a concise summary highlighting major topics and conclusions.",
    "Write a short overview focusing on the main discussions and outcomes of the dialogue.",
    "Provide a succinct summary that captures the essence and key outcomes of the dialogue.",
    "Summarize the dialogue by focusing on critical points and speaker contributions.",
    "Draft a concise summary that includes main topics and any decisions reached.",
    "Outline the primary points and conclusions discussed in the dialogue.",
    "Generate a summary that distills the dialogue into its most important discussions and decisions.",
    "Compose a brief summary emphasizing the key topics and decisions within the dialogue.",
    "Construct a concise encapsulation of the dialogue, noting significant topics and outcomes.",
    "Develop a summary that succinctly captures the essential points and results of the dialogue.",
    "Formulate a summary that condenses the main discussions and resolutions of the dialogue.",
    "Produce a short summary detailing the primary discussions and decisions from the dialogue.",
    "Prepare a concise overview of the dialogue, highlighting central themes and decisions.",
    "Craft a summary that focuses on the pivotal points and conclusions of the dialogue.",
    "Assemble a brief recap of the dialogue, emphasizing important topics and outcomes.",
    "Forge a succinct narrative summarizing the dialogue’s main points and resolutions.",
    "Synthesize the dialogue into a concise summary, focusing on key discussions and decisions.",
    "Compile a concise report of the dialogue, spotlighting major topics and outcomes.",
    "Create a streamlined summary that captures the core discussions and decisions of the dialogue."
]


def map_fn(sample):

    port = random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )

    # Dialog summarization
    text = sample['answer']['text']

    TEMPLATE = """\
        [Speech Transcription]
        {context}

        [Task]
        You are given one dialogue. Please summarize the main points discussed, focusing on key topics.
        Mention the speakers by name and highlight any specific contributions or statements they made that are critical to understanding the dialogue.
        Ensure your summary captures the essence and provides a clear overview of the dialogue without introducing any assumptions or interpretations not evident in the transcription.

        Format your response as follows:
        Summary: (Provide a concise summary of the dialogue here.)
        """

    format_sample = TEMPLATE.format(context=text)

    chat_response = client.chat.completions.create(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[{"role": "user", "content": format_sample}]
    )

    llm_output = chat_response.choices[0].message.content

    if "Summary:" in llm_output:
        llm_output = llm_output.split('Summary:')[1].strip()
    else:
        llm_output = "Template not matched."

    instruction = {'text': random.choice(
        candidate_instructions), 'audio': None, }
    answer = {'text': llm_output, 'audio': None, }

    new_sample = {
        'instruction': instruction,
        'answer': answer,
        'other_attributes': {'transcription': sample['answer']['text']}
    }

    return new_sample


def filter_fn(example):
    return example['answer']['text'].strip() not in ['Template not matched.', '']


def ds_generation(split, num_proc=32):

    ds = load_from_disk(split)

    features = ds.features
    features['other_attributes'] = {"transcription": Value(dtype='string')}

    ds = ds.filter(
        lambda x: len(x['answer']['text'].strip().split()) > 8,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "filter",
    )

    ds = ds.map(
        map_fn,
        features          = features,
        batched           = False,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "DS Generation",
    )

    ds = ds.filter(
        filter_fn,
        batch_size=1,
        writer_batch_size=1,
        num_proc=num_proc,
        desc="filter",
    )

    ds.save_to_disk(split.replace("ASR", "DS"), num_proc=4)


def main(pattern):
    splits = glob(pattern)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("ASR", "DS")):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        ds_generation(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
