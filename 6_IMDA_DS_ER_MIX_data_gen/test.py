


import os
import re
import random

from openai import OpenAI
from datasets import load_from_disk

import transformers

port = random.choice([5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007])
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{port}/v1",
)
    
prompt = """
    You're give 3 examples to convert the question and answer to a more proper format. You should provide the revised question and revised answer.
    Make sure the revised question is asking about the accent. 
    Please make sure the revised question indicate which speaker is referring to or both speakers.
    Do not output anything else except the revised question and revised answer.
    Avoid words like "from China" or "from India" and use "Chinese-Speaking community" or "Indian-Speaking community" instead.
    The speakers are from Singapore, so you can use "Singapore accent" or "Singaporean accent" in your answer by default.

    Example 1:
    Question: What's the ethnic group of the speaker based on their accent in the audio?
    First Speaker Information: First language is English. Spoken language is English.
    Second Speaker Information: First language is English. Spoken language is English.
    Number of speakers: 2
    
    Revised Question: What can you tell about the first speaker's accent?
    Revised Answer: The first speaker speaks good English with Singapore accent.

    Example 2:
    Question: So, what language does this person sound like they're from?
    First Speaker Information: First language is Malay. Spoken language is Malay,english.
    Second Speaker Information: First language is Malay. Spoken language is Malay,english.

    Revised Question: Can you describe the accent of the speaker?
    Revised Answer: Both speakers have a Malay accent. They are likely from the Malay-Speaking community of Singapore.

    Example 3:
    Question: So, what languages are these speakers speaking, based on their accents?
    First Speaker Information: First language is Chinese. Spoken language is Chinese,english.
    Second Speaker Information: First language is Chinese. Spoken language is Chinese,english.

    Revised Question: Can you infer the speakers background from their accents?
    Revised Answer: Both speakers are likely from the Chinese-Speaking community. Their spoken language is Chinese and English.


    You task now.
    Question: {question}
    First Speaker Information: First language is {first_speaker_first_language}. Spoken language is {first_speaker_spoken_language}.
    Second Speaker Information: First language is {second_speaker_first_language}. Spoken language is {second_speaker_spoken_language}.

    Please output Revised Question and Revised Answer.
"""

original_instruction = "What is the ethnic origin of the second speaker in the dialogue?"
original_answer = "Based on the accent, the ethnic group of the second speaker is Chinese."

first_speaker_first_language   = "English"
first_speaker_spoken_language  = "English, Chinese"
second_speaker_first_language  = "Chinese"
second_speaker_spoken_language = "English, Chinese"

input_message = prompt.format(
    question                       = original_instruction,
    first_speaker_first_language   = first_speaker_first_language,
    first_speaker_spoken_language  = first_speaker_spoken_language,
    second_speaker_first_language  = second_speaker_first_language,
    second_speaker_spoken_language = second_speaker_spoken_language
)

messages = [{"role": "user", "content": input_message},]

chat_response = client.chat.completions.create(
    model    = client.models.list().data[0].id,
    messages = messages
    )

transcription=chat_response.choices[0].message.content
print(transcription)

breakpoint()
    

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-70B-Instruct", device_map="auto", use_fast=False, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

templated_sample = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors        = "pt",
    tokenize              = False,
)

# breakpoint()

models = client.models.list()
model = models.data[0].id

completion = client.completions.create(
    model      = model,
    prompt     = templated_sample,
    max_tokens = 512,
    n          = 1,
)

output = completion.choices[0].text.strip()
print(output)


breakpoint()

print("Done")

