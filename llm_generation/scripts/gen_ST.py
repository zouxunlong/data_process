
import random
import fire
from datasets import load_from_disk, Value, concatenate_datasets
from openai import OpenAI
from glob import glob
import os


candidate_instructions = [
    "Please translate the speech into {language}.",
    "Convert the text of this speech into {language}.",
    "Render the contents of the speech given into {language}.",
    "Translate the speech from the video recording into {language}.",
    "Translate the entire speech into {language} for our records.",
    "Translate the speech, ensuring cultural nuances are preserved in {language}.",
    "Please translate the speech into {language}, including any idiomatic expressions.",
    "Translate the formal speech into {language} with attention to formal language usage.",
    "Can you translate this speech into {language} for me?",
    "Could you help convert this speech into {language}?",
    "Mind translating this quick speech into {language}?",
    "Hey, can you turn this speech into {language}?",
    "Need this speech in {language}—can you help?",
    "Can you make this speech understandable in {language}?",
    "Could you switch this speech to {language}?",
    "Can you flip this speech into {language} for me?",
    "Would you mind translating this speech into {language}?",
    "Can you help me get this speech into {language}?",
    "Got a sec to translate this speech into {language}?",
    "Hey, can you work your magic and translate this into {language}?",
    "Can you get this speech translated into {language}?",
    "Can you redo this speech in {language}?",
    "Could you rewrite this speech in {language}?",
    "Any chance you can convert this speech to {language}?",
    "Can you craft this speech into {language}?",
    "Can you shift this speech into {language}?",
    "Can you tweak this speech into {language}?",
    "Could you help by translating this speech into {language}?"
]


def map_fn(sample, language):

    port = random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    
    # Question generation

    text = sample['answer']['text']

    QUESTION_TEMPLATE = """\
        [Speech Transcription]
        {context}

        [Task]
        You are given one speech transcription. Please translate the transcription into {language}.
        Do not output anything other than the {language} translation.
        """

    prompt_sample = QUESTION_TEMPLATE.format(context=text, language=language)

    chat_response = client.chat.completions.create(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[{"role": "user", "content": prompt_sample},]
        )

    translation=chat_response.choices[0].message.content
    
    return {
        'instruction': {'text': random.choice(candidate_instructions).format(language=language), 'audio': None},
        'answer': {'text': translation, 'audio': None},
        'other_attributes': {'transcription': sample['answer']['text']}
    }

def filter_fn(example):
    return example['answer']['text'].strip() not in ['Template not matched.', '']

def st_generation(split, language, lang_code, num_proc=128):

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
        fn_kwargs={'language': language},
        features          = features,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc="ST Generation for {}".format(split),
    )

    ds = ds.filter(
        filter_fn,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "filter",
    )

    ds.save_to_disk(split.replace("/ASR/", "/ST/").replace("_ASR_", "_en_{}_ST_".format(lang_code)), num_proc=4)


def main(pattern, language, lang_code):
    splits = glob(pattern)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("/ASR/", "/ST/").replace("_ASR_", "_en_{}_ST_".format(lang_code))):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        st_generation(split, language, lang_code)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
