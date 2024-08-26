
from glob import glob
import os
import random
import fire
from datasets import load_from_disk
from openai import OpenAI


def map_fn(batch):
    
    responses=[]

    for instruction, answer in zip(batch["instruction"], batch["answer"]):

        PROMPT_TEMPLATE = """\
            [Question]
            {question}

            [Ground Truth Reference]
            {reference}

            [System]
            You're given a question and a template based groundtruth reference based on an audio clip.
            Please rewrite the question and the groundtruth reference to be more natural and diverse, either in a formal tongue or in a informal converse way.
            Please strictly make sure you only re-write the question and answer in one of the above style and only generate one pair of Question and Answer.
            Please strictly make sure they convey the same meaning and sounds nature as a response to the question based on the audio clip.
            Additional, provide a brief explanation of the rationale behind your answer.

            Format your response as follows:
            Explanation: (Briefly explain the rationale behind your question and answer.)
            Question: (your rephrased question here)
            Answer: (your rephrased answer here)"""

        prompt_sample = PROMPT_TEMPLATE.format(question=instruction["text"], reference=answer["text"])

        port=random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006])
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

        responses.append(chat_response.choices[0].message.content)

    questions = [sample.split("Question: ")[1].split("\n")[0].strip() if 'Question: ' in sample else "No Question Found" for sample in responses]
    answers = [sample.split("Answer: ")[1].split("\n")[0].strip() if 'Answer: ' in sample else "No Answer Found" for sample in responses]
    
    batch["instruction"]=[{"audio":None, "text":question} for question in questions]
    batch["answer"]=[{"audio":None, "text":answer} for answer in answers]

    return batch


def qa_rephrase(split):

    data = load_from_disk(split)
    data = data.map(
        map_fn,
        features          = data.features,
        batched           = True,
        batch_size        = 1,
        num_proc          = 128,
        writer_batch_size = 1,
        desc              = "QA re-write for {}".format(split.split("/IMDA/")[-1]),
    )

    data.filter(lambda x: x['instruction']['text'] != 'No Question Found', num_proc=20)
    data.filter(lambda x: x['answer']['text'] != 'No Answer Found', num_proc=20)

    data.save_to_disk(split.replace("_v1", "_v2"), num_proc=4)



def main(pattern):
    splits = glob(pattern)
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("_v1", "_v2")):
            print("exists {}".format(split), flush=True)
            continue
        print("start {}".format(split), flush=True)
        qa_rephrase(split)
        print("complete {}".format(split), flush=True)

if __name__ == '__main__':
    fire.Fire(main)

