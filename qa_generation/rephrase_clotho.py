
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
            You're given the question and the groundtruth reference based on an audio clip.
            Please rewrite the groundtruth reference as one complete sentence.
            Please strictly make sure they convey the same meaning and sounds nature as a response to the question based on the audio clip.
            Additional, provide a brief explanation of the rationale behind your question.

            Format your response as follows:
            Explanation: (Briefly explain the rationale behind your question.)
            Answer: (your rephrased answer)"""

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

    answers = [sample.split("Answer: ")[1].split("\n")[0].strip() if 'Answer: ' in sample else "No Answer Found" for sample in responses]
    
    batch["answer"]=[{"audio":None, "text":answer} for answer in answers]
    

    return batch



def main():

    for split in ["test", "train", "validation"]:

        dataset = load_from_disk('/mnt/home/zoux/xunlong_working_repo/data_AQA/clotho_aqa/clotho_aqa.hf/{}'.format(split))
        dataset = dataset.map(
            map_fn,
            batched=True,
            batch_size=1,
            num_proc=64,
            writer_batch_size=1,
            desc="Answer rephrase",
        )
        
        dataset.filter(lambda x: x['answer']['text'] != 'No Answer Found', num_proc=20)

        dataset.save_to_disk('/mnt/home/zoux/xunlong_working_repo/data_AQA/clotho_aqa/clotho_aqa_v1/{}'.format(split), num_proc=4)


if __name__ == '__main__':
    fire.Fire(main)


