
import random
import fire
from datasets import load_from_disk, Value
from openai import OpenAI
from glob import glob
import os


def map_fn(batch_samples):


    # Question generation
    generated_questions = []
    for sample in batch_samples['answer']:

        text = sample['text']

        QUESTION_TEMPLATE = """\
            [Speech Transcription]
            {context}

            [Task]
            You are given one speech transcription. Please generate two factual questions related to the transcription.
            Make sure that the transcription has enough information to provide the answer to these questions.
            If the question is about a specific speaker, please mention the speaker's name in the question.
            Do not output 'according to the transcription'.
            Additional, provide a brief explanation of the rationale behind your question.

            Format your response as follows:
            Question: (Write your question here)
            Explanation: (Briefly explain the rationale behind your question.)
            Question: (Write your question here)
            Explanation: (Briefly explain the rationale behind your question.)
            """

        prompt_sample = QUESTION_TEMPLATE.format(context=text)

        port=random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )

        chat_response = client.chat.completions.create(
            model="/mnt/home/zoux/models/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "user", "content": prompt_sample},
            ]
        )

        generated_questions.append(chat_response.choices[0].message.content)

    generated_questions = [[item.split('Explanation: ')[0].strip() for item in sample.split('Question: ')[1:]] if 'Question: ' in sample and len(sample.split('Question: '))==3 else ["No Question Found"]*2 for sample in generated_questions]


    # Generate the answer
    generated_answers = []
    for questions, sample in zip(generated_questions, batch_samples['answer']):

        text = sample['text']

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

        for question in questions:
            format_sample = ANSWER_TEMPLATE.format(context=text, question=question)

            port=random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
            client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1",
            )
            
            chat_response = client.chat.completions.create(
                model="/mnt/home/zoux/models/Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": format_sample},
                ]
            )
            generated_answers.append(chat_response.choices[0].message.content)

    generated_answers = [sample.split('Answer: ')[1].strip() if 'Answer: ' in sample else "No Answer Found" for sample in generated_answers]


    instructions     = [{'text': question, 'audio': None} for questions in generated_questions for question in questions]
    answers          = [{'text': answer, 'audio': None} for answer in generated_answers]
    other_attributes = [{'transcription': sample['text']} for _sample in batch_samples['answer'] for sample in [_sample]*2]
    contexts         = [context for _context in batch_samples['context'] for context in [_context]*2]

    new_batch = {
        'context'         : contexts,
        'instruction'     : instructions,
        'answer'          : answers,
        'other_attributes': other_attributes
    }
    return new_batch


def qa_generation(split):

    ds = load_from_disk(split)

    features = ds.features
    features['other_attributes'] = {"transcription": Value(dtype='string')}

    ds = ds.map(
        map_fn,
        features          = features,
        batched           = True,
        batch_size        = 1,
        num_proc          = 128,
        writer_batch_size = 1,
        desc              = "QA Generation for {}".format(split.split("/IMDA/")[-1]),
    )

    ds = ds.filter(lambda x: x['instruction']['text'] != 'No Question Found' and x['answer']['text'] != 'No Answer Found',
                   batch_size=1,
                   writer_batch_size=1,
                   num_proc=20
                   )

    ds = ds.shuffle()

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
