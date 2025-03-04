

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

        You're give 5 examples to generate a question and answer based on the Meta facts of the speakers in an audio clip.
        Make sure the generated question is asking about the speaker's accent. 
        Please make sure the question indicate which speaker is referring to or both speakers.
        Do not output anything else except the question and answer.
        Avoid words like "from China" or "from India" and use "Chinese-Speaking community" or "Indian-Speaking community" instead.
        The speakers are from Singapore, so you can use "Singapore accent" or "Singaporean accent" in your answer by default.

        Example 1:
        First Speaker Information: First language is English.
        Second Speaker Information: First language is Tamil.
        Number of speakers: 2

        Question: What can you tell about the first speaker's accent?
        Answer: The first speaker speaks good English with Singapore accent.

        Example 2:
        First Speaker Information: First language is Malay.
        Second Speaker Information: First language is Malay.
        Number of speakers: 2

        Question: Can you describe the accent of the speakers?
        Answer: Both speakers have a Malay accent. They are likely from the Malay-Speaking community of Singapore.

        Example 3:
        First Speaker Information: First language is Chinese.
        Second Speaker Information: First language is Cantonese.
        Number of speakers: 2

        Question: Can you infer the speakers background from their accents?
        Answer: Both speakers are likely from Singapore's Chinese-Speaking community. Their first language is Chinese.

        Example 4:
        Speaker Information: First language is Tamil.
        Number of speakers: 1

        Question: What can you tell about the speaker's accent?
        Answer: The speaker have a Indian accent. They are likely from the Indian-Speaking community of Singapore.

        Example 5:
        Speaker Information: First language is Hokkien.
        Number of speakers: 1

        Question: Can you describe the accent characteristics of the male speaker?
        Answer: The speaker have a Hokkien accent. They are likely from the Chinese-Speaking community of Singapore.


        You task now.
        {speaker_information}

        Please output Question and Answer as format below:
        Question: (Write the question here)
        Answer: (Write the answer here)
    """


    if "speaker" in batch['other_attributes'][0]:
        speaker=batch["other_attributes"][0]["speaker"]
        if "first_language" in speaker:
            first_language = speaker['first_language']
        elif "ethnic_group" in speaker:
            first_language = speaker['ethnic_group']
        else:
            batch['instruction'][0]['text'] = "No match found!!!!"
            batch['answer'][0]['text']      = "No match found!!!!"
            return batch
        speaker_information=f"""Speaker Information: First language is {first_language}.
        Number of speakers: 1"""
    else:
        speaker1        = batch["other_attributes"][0]["speaker1"]
        speaker2        = batch["other_attributes"][0]["speaker2"]
        first_language1 = speaker1['first_language']
        first_language2 = speaker2['first_language']
        transcription   = batch["answer"][0]["text"]
        if re.search("<Speaker2>: ", transcription):
            speaker_information=f"""First Speaker Information: First language is {first_language1}.
            Second Speaker Information: First language is {first_language2}.
            Number of speakers: 2"""
        else:
            speaker_information=f"""Speaker Information: First language is {first_language1}.
            Number of speakers: 1"""



    input_message = prompt.format(speaker_information = speaker_information)
    port = random.choice([5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    models = client.models.list()
    model = models.data[0].id
    chat_response = client.chat.completions.create(
        model      = model,
        messages   = [{"role": "user", "content": input_message},],
        max_tokens = 512,
        n          = 1,
        )
    output=chat_response.choices[0].message.content

    # Use regex to extract the Revised Question and Revised Answer
    pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*)"
    match = re.search(pattern, output, re.DOTALL)

    if match:
        revised_question = match.group(1).strip()
        revised_answer = match.group(2).strip()

    else:
        revised_question = "No match found!!!!"
        revised_answer = "No match found!!!!"

    batch['instruction'][0]['text'] = revised_question
    batch['answer'][0]['text']      = revised_answer
    
    return batch


def build(ROOT_PATH, DATASET_NAME):

    output_path = os.path.join(ROOT_PATH.replace("/ASR", "/PQA"), DATASET_NAME.replace("_ASR_","_AR_"))
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists")
        return

    data = load_from_disk(os.path.join(ROOT_PATH, DATASET_NAME))
    data = data.map(map_fn,
                    num_proc          = 224,
                    batched           = True,
                    batch_size        = 1,
                    writer_batch_size = 1,
                    features          = data.features,
                    desc              = f"map AR/{DATASET_NAME}",
                    )

    data = data.filter(lambda x: x['instruction']['text'] != "No match found!!!!",
                        num_proc          = 224,
                        batch_size        = 1,
                        writer_batch_size = 1,
                        desc              = f"filter AR/{DATASET_NAME}",
                        )

    data.save_to_disk(output_path, num_proc=10)


def main(split="all", dataset="all"):
    for ROOT_PATH in ['/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/ASR',
                      '/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/train/ASR']:

        for DATASET_NAME in [
            'IMDA_PART1_ASR_v4',
            'IMDA_PART2_ASR_v4',
            'IMDA_PART3_ASR_v4',
            'IMDA_PART4_ASR_v4',
            'IMDA_PART5_ASR_v4',

            'IMDA_PART3_30_ASR_v4',
            'IMDA_PART4_30_ASR_v4',
            'IMDA_PART5_30_ASR_v4',

            'IMDA_PART3_60_ASR_v4',
            'IMDA_PART4_60_ASR_v4',
            'IMDA_PART5_60_ASR_v4',

            'IMDA_PART3_120_ASR_v4',
            'IMDA_PART4_120_ASR_v4',
            'IMDA_PART5_120_ASR_v4',

            'IMDA_PART3_300_ASR_v4',
            'IMDA_PART4_300_ASR_v4',
            'IMDA_PART5_300_ASR_v4']:
            if (split == "all" or split in ROOT_PATH) and (dataset == "all" or dataset in DATASET_NAME):
                print(f"Processing {ROOT_PATH}/{DATASET_NAME}", flush=True)
                build(ROOT_PATH, DATASET_NAME)


if __name__ == "__main__":
    fire.Fire(main)