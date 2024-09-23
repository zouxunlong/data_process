
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
        

        port = random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006])
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )
        

        chat_response = client.chat.completions.create(
            model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            messages=[
                {"role": "user", "content": prompt_sample},
            ]
        )
        

        responses.append(chat_response.choices[0].message.content)

    answers = [sample.split("Answer: ")[1].split("\n")[0].strip() if 'Answer: ' in sample else "Template not matched." for sample in responses]
    
    batch["answer"]=[{"audio":None, "text":answer} for answer in answers]
    

    return batch



def main(num_proc=64):

    for split in ["test", "train", "validation"]:

        dataset = load_from_disk('/mnt/home/zoux/workspaces/xunlong_working_repo/data_ASQA/clotho_aqa/clotho_aqa.schemed/{}'.format(split))
        dataset = dataset.map(
            map_fn,
            batched=True,
            batch_size=1,
            num_proc=num_proc,
            writer_batch_size=1,
            desc="Answer rephrase",
        )
        
        dataset=dataset.filter(lambda x: x['answer']['text'] != 'Template not matched.', num_proc=64)

        dataset.save_to_disk('/mnt/home/zoux/workspaces/xunlong_working_repo/data_ASQA/clotho_aqa/clotho_ASQA_v2/{}'.format(split), num_proc=4)


if __name__ == '__main__':
    fire.Fire(main)


