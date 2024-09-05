
import random
import fire
from datasets import load_from_disk, Value
from openai import OpenAI
from glob import glob
import os


candidate_instructions = [
    "Please translate the speech into Chinese.",
    "Convert the text of this speech into Chinese.",
    "Render the contents of the speech given into Chinese.",
    "Translate the speech from the video recording into Chinese.",
    "Translate the entire speech into Chinese for our records.",
    "Translate the speech, ensuring cultural nuances are preserved in Chinese.",
    "Please translate the speech into Chinese, including any idiomatic expressions.",
    "Translate the formal speech into Chinese with attention to formal language usage.",
    "Can you translate this speech into Chinese for me?",
    "Could you help convert this speech into Chinese?",
    "Mind translating this quick speech into Chinese?",
    "Hey, can you turn this speech into Chinese?",
    "Need this speech in Chinese—can you help?",
    "Can you make this speech understandable in Chinese?",
    "Could you switch this speech to Chinese?",
    "Can you flip this speech into Chinese for me?",
    "Would you mind translating this speech into Chinese?",
    "Can you help me get this speech into Chinese?",
    "Got a sec to translate this speech into Chinese?",
    "Hey, can you work your magic and translate this into Chinese?",
    "Can you get this speech translated into Chinese?",
    "Can you redo this speech in Chinese?",
    "Could you rewrite this speech in Chinese?",
    "Any chance you can convert this speech to Chinese?",
    "Can you craft this speech into Chinese?",
    "Can you shift this speech into Chinese?",
    "Can you tweak this speech into Chinese?",
    "Could you help by translating this speech into Chinese?"
]


def map_fn(batch_samples):

    # Question generation
    generated_translations = []
    for sample in batch_samples['answer']:

        text = sample['text']

        QUESTION_TEMPLATE = """\
            [Speech Transcription]
            {context}

            [Task]
            You are given one speech transcription. Please translate the transcription into Chinese.
            Do not output anything other than the Chinese translation.
            """

        prompt_sample = QUESTION_TEMPLATE.format(context=text)

        port = random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )

        chat_response = client.chat.completions.create(
            model="casperhansen/llama-3-70b-instruct-awq",
            messages=[
                {"role": "user", "content": prompt_sample},
            ]
        )


        generated_translations.append(chat_response.choices[0].message.content)


    instructions = [{'text': random.choice(candidate_instructions), 'audio': None} for translation in generated_translations]
    answers = [{'text': translation, 'audio': None} for translation in generated_translations]
    other_attributes = [{'transcription': sample['text']} for sample in batch_samples['answer']]
    breakpoint()
    
    new_batch = {
        'instruction': instructions,
        'answer': answers,
        'other_attributes': other_attributes
    }
    return new_batch


def st_generation(split):

    ds = load_from_disk(split)

    features = ds.features
    features['other_attributes'] = {"transcription": Value(dtype='string')}

    ds = ds.map(
        map_fn,
        features=features,
        batched=True,
        batch_size=1,
        num_proc=1,
        writer_batch_size=1,
        desc="ST Generation for {}".format(split),
    )

    ds = ds.filter(lambda x: x['instruction']['text']
                   != 'No Question Found', num_proc=20)
    ds = ds.filter(lambda x: x['answer']['text'] !=
                   'No Answer Found', num_proc=20)

    ds = ds.shuffle()

    ds.save_to_disk(split.replace("ASR", "ST"), num_proc=4)


def main(pattern):
    splits = glob(pattern)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("ASR", "ST")):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        st_generation(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
