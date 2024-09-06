
import random
import fire
from datasets import load_from_disk, Value
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

    context = sample['answer']["text"]
    QUESTION_TEMPLATE = """\
        [Speech Transcription]
        {context}

        [Task]
        You are given one speech transcription. Please generate a factual question related to the transcription.
        Make sure that the transcription has enough information to provide the answer to the question.
        If the question is about a specific speaker, please mention the speaker's name in the question.
        Do not output 'according to the transcription'.
        Additional, provide a brief explanation of the rationale behind your generated question.

        Format your response as follows:
        Question: (Write your question here)
        Explanation: (Briefly explain the rationale behind your question.)
        """

    prompt_sample = QUESTION_TEMPLATE.format(context=context)

    chat_response = client.chat.completions.create(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[{"role": "user", "content": prompt_sample},])

    res_content = chat_response.choices[0].message.content

    if 'Question: ' in res_content and len(res_content.split('Question: ')) == 2:
        question = res_content.split('Question: ')[1].split('Explanation: ')[0].strip()
    else:
        question = "Template not matched."

    # Generate the answer
    ANSWER_TEMPLATE = """\
        [Speech Transcription]
        {context}

        [Question]
        {question}

        [Task]
        Your task is to provide a concrete answer to the question, based on the dialogue transcription given. 
        Ensure that your answer is a direct answer to the question.
        Do not output 'according to the transcription'.
        Keep your response within one short sentence.

        Please format your response as follows:
        Answer: (Write your answer here)
        """

    if question in ["Template not matched.", ""]:
        answer = "Template not matched."
    else:
        format_sample = ANSWER_TEMPLATE.format(
            context=context, question=question)

        chat_response = client.chat.completions.create(
            model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            messages=[{"role": "user", "content": format_sample},]
        )

        res_content = chat_response.choices[0].message.content

        if 'Answer: ' in res_content:
            answer = res_content.split('Answer: ')[1].strip()
        else:
            answer = "Template not matched."

    return {
        'instruction'     : {'text': question, 'audio': None},
        'answer'          : {'text': answer, 'audio': None},
        'other_attributes': {'transcription': context}
    }


def qa_generation(split, num_proc=64):

    ds = load_from_disk(split)

    features = ds.features
    features['other_attributes'] = {"transcription": Value(dtype='string')}

    ds = ds.map(
        map_fn,
        features          = features,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "QA Generation",
    )

    def filter_fn(example):
        return example['answer']['text'].strip() not in ['Template not matched.', '']

    ds = ds.filter(
        filter_fn,
        batch_size        = 1,
        writer_batch_size = 1,
        num_proc          = num_proc,
        desc              = "filter",
    )

    ds.save_to_disk(split.replace("ASR", "SQA"), num_proc=4)


def main(pattern):
    splits = glob(pattern)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("ASR", "SQA")):
            print("complete {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        qa_generation(split)
        print("complete {}".format(split), flush=True)


if __name__ == '__main__':
    fire.Fire(main)
