

import os
import re
import random

from openai import OpenAI
from datasets import load_from_disk

import transformers


# os.environ["HF_HOME"] = "~/scratch/huggingface"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# HF_HOME=~/scratch/huggingface HF_ENDPOINT=https://hf-mirror.com python imda_accent_reformat.py 2>&1 | tee imda_accent_reformat.log


ROOT_PATH = '/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/test/ASR'


for DATASET_NAME in [
    'IMDA_PART3_30_ASR_v2',
    'IMDA_PART4_30_ASR_v2',
    'IMDA_PART5_30_ASR_v2',

    'IMDA_PART3_60_ASR_v2',
    'IMDA_PART4_60_ASR_v2',
    'IMDA_PART5_60_ASR_v2',

    'IMDA_PART3_120_ASR_v2',
    'IMDA_PART4_120_ASR_v2',
    'IMDA_PART5_120_ASR_v2',
]:

    data = load_from_disk(os.path.join(ROOT_PATH, DATASET_NAME))

    MODEL_PORT = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-70B-Instruct", device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    def reformat_instruction_answer(batch):

        prompt = """

            You're give 5 examples to generate a question and answer based on the Meta information of the speakers in an audio clip.
            Make sure the generated question is asking about the gender and accent. 
            Please make sure the generated question indicate which speaker is referring to or both speakers.
            Do not output anything else except the generated question and answer.
            Avoid words like "from China" or "from India" and use "Chinese-Speaking community" or "Indian-Speaking community" instead.
            The speakers are from Singapore, so you can use "Singapore accent" or "Singaporean accent" in your answer by default.

            Example 1:
            First Speaker Information: Male, First language is Chinese.
            Second Speaker Information: Female, First language is English.
            Number of speakers: 2

            Question: Based on the audio provided, could you identify the gender and accent of the first speaker?
            Answer: The first speaker is male and speaks English fluently with a Singaporean accent, along with a hint of a Chinese accent.

            Example 2:
            First Speaker Information: Female, First language is Chinese.
            Second Speaker Information: Male, First language is Tamil.
            Number of speakers: 2

            Question: Based on the audio provided, could you identify the gender and accent of the second speaker?
            Answer: The second speaker is male and most likely from an Indian-Speaking community in Singapore.

            Example 3:
            First Speaker Information: Male, First language is Chinese.
            Second Speaker Information: Female, First language is Malay.
            Number of speakers: 2

            Question: Can you describe the gender and accent of the speakers?
            Answer: The first speaker is male and have a Chinese accent, the second speaker is Female and have a Malay accent.

            Example 4:
            First Speaker Information: Male, First language is Tamil.
            Second Speaker Information: Male, First language is Tamil.
            Number of speakers: 2

            Question: Can you infer the speakers' gender and background from their voice and accents?
            Answer: Yes, both speakers are male and have a Tamil accent, most likely they are from Indian-speaking community in Singapore.

            Example 5:
            Speaker Information: Female, First language is English.
            Number of speakers: 1

            Question: Can you infer the speaker's gender and background from the voice and accents?
            Answer: Yes, The speaker is female and speaks English fluently, with an accent of Singaporean.


            You task now.
            {speaker_information}

            Please output Question and Answer as format below:
            Question: (Write the question here)
            Answer: (Write the answer here)
        """


        first_speaker=batch["other_attributes"][0]["speaker1"]
        second_speaker=batch["other_attributes"][0]["speaker2"]

        first_gender="female" if first_speaker["gender"] in ["F", "Female"] else "male"
        second_gender="female" if second_speaker["gender"] in ["F", "Female"] else "male"

        first_speaker_first_language   = first_speaker['first_language']
        second_speaker_first_language  = second_speaker['first_language']

        transcription=batch["answer"][0]["text"]
        if re.search("\n", transcription): 
            speaker_information=f"""First Speaker Information: {first_gender}, First language is {first_speaker_first_language}.
            Second Speaker Information: {second_gender}, First language is {second_speaker_first_language}.
            Number of speakers: 2"""
        else: 
            speaker_information=f"""Speaker Information: {first_gender}, First language is {first_speaker_first_language}.
            Number of speakers: 1"""

        input_message = prompt.format(speaker_information = speaker_information)

        messages = [
            {"role": "user", "content": input_message},
        ]

        templated_sample = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors        = "pt",
            tokenize              = False,
        )

        # Model
        port            = random.choice(MODEL_PORT)
        openai_api_key  = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        client          = OpenAI(
            api_key  = openai_api_key,
            base_url = openai_api_base,
        )

        models = client.models.list()
        model = models.data[0].id

        completion = client.completions.create(
            model      = model,
            prompt     = templated_sample,
            max_tokens = 512,
            n          = 1,
        )

        output = completion.choices[0].text.strip()

        # Use regex to extract the Revised Question and Revised Answer
        pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*)"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            revised_question = match.group(1).strip()
            revised_answer = match.group(2).strip()

        else:

            revised_question = "No match found!!!!"
            revised_answer = "No match found!!!!"

            print("No match found!!!!")
            print("output:", output)

        batch['instruction'][0]['text']               = revised_question
        batch['answer'][0]['text']                    = revised_answer
        batch['other_attributes'][0]['transcription'] = transcription
        
        return batch

    data = data.map(reformat_instruction_answer,
                    num_proc=256,
                    batched=True,
                    batch_size=1,
                    writer_batch_size=1,
                    )

    data = data.filter(lambda x: x['instruction']['text'] != "No match found!!!!",
                        num_proc=256,
                        batch_size=1,
                        writer_batch_size=1,
                        )

    data.save_to_disk(os.path.join(ROOT_PATH.replace("/ASR", "/Paralingual"), DATASET_NAME.replace("_ASR_","_MIX_")))





ROOT_PATH = '/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/train/ASR'


for DATASET_NAME in [
    'IMDA_PART3_30_ASR_v2',
    'IMDA_PART4_30_ASR_v2',
    'IMDA_PART5_30_ASR_v2',

    'IMDA_PART3_60_ASR_v2',
    'IMDA_PART4_60_ASR_v2',
    'IMDA_PART5_60_ASR_v2',

    'IMDA_PART3_120_ASR_v2',
    'IMDA_PART4_120_ASR_v2',
    'IMDA_PART5_120_ASR_v2',
]:

    data = load_from_disk(os.path.join(ROOT_PATH, DATASET_NAME))

    MODEL_PORT = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-70B-Instruct", device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    def reformat_instruction_answer(batch):

        prompt = """

            You're give 5 examples to generate a question and answer based on the Meta information of the speakers in an audio clip.
            Make sure the generated question is asking about the gender and accent. 
            Please make sure the generated question indicate which speaker is referring to or both speakers.
            Do not output anything else except the generated question and answer.
            Avoid words like "from China" or "from India" and use "Chinese-Speaking community" or "Indian-Speaking community" instead.
            The speakers are from Singapore, so you can use "Singapore accent" or "Singaporean accent" in your answer by default.

            Example 1:
            First Speaker Information: Male, First language is Chinese.
            Second Speaker Information: Female, First language is English.
            Number of speakers: 2

            Question: Based on the audio provided, could you identify the gender and accent of the first speaker?
            Answer: The first speaker is male and speaks English fluently with a Singaporean accent, along with a hint of a Chinese accent.

            Example 2:
            First Speaker Information: Female, First language is Chinese.
            Second Speaker Information: Male, First language is Tamil.
            Number of speakers: 2

            Question: Based on the audio provided, could you identify the gender and accent of the second speaker?
            Answer: The second speaker is male and most likely from an Indian-Speaking community in Singapore.

            Example 3:
            First Speaker Information: Male, First language is Chinese.
            Second Speaker Information: Female, First language is Malay.
            Number of speakers: 2

            Question: Can you describe the gender and accent of the speakers?
            Answer: The first speaker is male and have a Chinese accent, the second speaker is Female and have a Malay accent.

            Example 4:
            First Speaker Information: Male, First language is Tamil.
            Second Speaker Information: Male, First language is Tamil.
            Number of speakers: 2

            Question: Can you infer the speakers' gender and background from their voice and accents?
            Answer: Yes, both speakers are male and have a Tamil accent, most likely they are from Indian-speaking community in Singapore.

            Example 5:
            Speaker Information: Female, First language is English.
            Number of speakers: 1

            Question: Can you infer the speaker's gender and background from the voice and accents?
            Answer: Yes, The speaker is female and speaks English fluently, with an accent of Singaporean.


            You task now.
            {speaker_information}

            Please output Question and Answer as format below:
            Question: (Write the question here)
            Answer: (Write the answer here)
        """


        first_speaker=batch["other_attributes"][0]["speaker1"]
        second_speaker=batch["other_attributes"][0]["speaker2"]

        first_gender="female" if first_speaker["gender"] in ["F", "Female"] else "male"
        second_gender="female" if second_speaker["gender"] in ["F", "Female"] else "male"

        first_speaker_first_language   = first_speaker['first_language']
        second_speaker_first_language  = second_speaker['first_language']

        transcription=batch["answer"][0]["text"]
        if re.search("\n", transcription): 
            speaker_information=f"""First Speaker Information: {first_gender}, First language is {first_speaker_first_language}.
            Second Speaker Information: {second_gender}, First language is {second_speaker_first_language}.
            Number of speakers: 2"""
        else: 
            speaker_information=f"""Speaker Information: {first_gender}, First language is {first_speaker_first_language}.
            Number of speakers: 1"""

        input_message = prompt.format(speaker_information = speaker_information)

        messages = [
            {"role": "user", "content": input_message},
        ]

        templated_sample = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors        = "pt",
            tokenize              = False,
        )

        # Model
        port            = random.choice(MODEL_PORT)
        openai_api_key  = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        client          = OpenAI(
            api_key  = openai_api_key,
            base_url = openai_api_base,
        )

        models = client.models.list()
        model = models.data[0].id

        completion = client.completions.create(
            model      = model,
            prompt     = templated_sample,
            max_tokens = 512,
            n          = 1,
        )

        output = completion.choices[0].text.strip()

        # Use regex to extract the Revised Question and Revised Answer
        pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*)"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            revised_question = match.group(1).strip()
            revised_answer = match.group(2).strip()

        else:

            revised_question = "No match found!!!!"
            revised_answer = "No match found!!!!"

            print("No match found!!!!")
            print("output:", output)

        batch['instruction'][0]['text']               = revised_question
        batch['answer'][0]['text']                    = revised_answer
        batch['other_attributes'][0]['transcription'] = transcription
        
        return batch

    data = data.map(reformat_instruction_answer,
                    num_proc=256,
                    batched=True,
                    batch_size=1,
                    writer_batch_size=1,
                    )

    data = data.filter(lambda x: x['instruction']['text'] != "No match found!!!!",
                        num_proc=256,
                        batch_size=1,
                        writer_batch_size=1,
                        )

    data.save_to_disk(os.path.join(ROOT_PATH.replace("/ASR", "/Paralingual"), DATASET_NAME.replace("_ASR_","_MIX_")))


