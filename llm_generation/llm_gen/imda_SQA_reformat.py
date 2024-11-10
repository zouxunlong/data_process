

import os
import re
import random
import fire
from openai import OpenAI
from datasets import load_from_disk


# os.environ["HF_HOME"] = "~/scratch/huggingface"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# HF_HOME=~/scratch/huggingface HF_ENDPOINT=https://hf-mirror.com python imda_accent_reformat.py 2>&1 | tee imda_accent_reformat.log


def map_fn(batch):

    prompt = """

        Task: You are given one speech transcription. Please generate a factual question related to the transcription, and the answer to the question.
        Make sure that the transcription has enough information to provide the answer to the question.
        If the question is about a specific speaker, please mention the speaker's name in the question.
        Do not output 'according to the transcription'.
        Additional, provide a brief explanation of the rationale behind your generated question.

        Transcription: {context}
        
        Format your response as follows:
        Explanation: (Briefly explain the rationale behind your question.)
        Question: (Write the question here)
        Answer: (Write the answer here)
    """

    transcription = batch["answer"][0]["text"]

    input_message = prompt.format(context=transcription)

    port = random.choice([5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    models = client.models.list()
    model = models.data[0].id
    chat_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_message},],
        max_tokens=512,
        n=1,
    )
    output = chat_response.choices[0].message.content

    # Use regex to extract the Revised Question and Revised Answer
    pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*)"
    match = re.search(pattern, output, re.DOTALL)

    if match:
        revised_question = match.group(1).strip()
        revised_answer = match.group(2).strip()
    else:
        revised_question = "No match found!!!!"
        revised_answer = "No match found!!!!"

        print("No match found!!!!", flush=True)
        print("output:", output, flush=True)

    batch['instruction'][0]['text'] = revised_question
    batch['answer'][0]['text'] = revised_answer
    batch['other_attributes'][0]['transcription'] = transcription
    return batch


def build(ROOT_PATH, DATASET_NAME):

    output_path = os.path.join(ROOT_PATH.replace("/ASR", "/SQA"), DATASET_NAME.replace("_ASR_", "_SQA_"))
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists")
        return

    data = load_from_disk(os.path.join(ROOT_PATH, DATASET_NAME))    
    data = data.map(map_fn,
                    num_proc=224,
                    batched=True,
                    batch_size=1,
                    writer_batch_size=1,
                    )

    data = data.filter(lambda x: x['instruction']['text'] not in ["No match found!!!!", ""],
                       num_proc          = 224,
                       batch_size        = 1,
                       writer_batch_size = 1,
                       )

    data.save_to_disk(output_path, num_proc=2)


def main(split="all", dataset="all"):
    for ROOT_PATH in ['/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes/test/ASR',
                      '/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes/train/ASR']:

        for DATASET_NAME in ['IMDA_PART3_30_ASR_v4',
                             'IMDA_PART4_30_ASR_v4',
                             'IMDA_PART5_30_ASR_v4',
                             'IMDA_PART6_30_ASR_v4',

                             'IMDA_PART3_60_ASR_v4',
                             'IMDA_PART4_60_ASR_v4',
                             'IMDA_PART5_60_ASR_v4',
                             'IMDA_PART6_60_ASR_v4',

                             'IMDA_PART3_120_ASR_v4',
                             'IMDA_PART4_120_ASR_v4',
                             'IMDA_PART5_120_ASR_v4',
                             'IMDA_PART6_120_ASR_v4'
                             ]:
            if (split == "all" or split in ROOT_PATH) and (dataset == "all" or dataset in DATASET_NAME):
                build(ROOT_PATH, DATASET_NAME)


if __name__ == "__main__":
    fire.Fire(main)