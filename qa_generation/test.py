from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch
from fire import Fire
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


PROMPT_TEMPLATE = """\
    [Question]
    {question}

    [Ground Truth Reference]
    {reference}

    [System]
    You're given the question and the groundtruth reference based on an audio recording.
    Please rephrase the question into five diffenrent style while strictly keeping the same query.
    Please rewrite the groundtruth reference as one complete sentence to answer the coresponding question.
    Please strictly make sure the answer convey the same meaning and sounds nature as an answer based on the audio recording.
    Your response should be formatted as follows:
    Q&As: [("question", "answer"), ...]"""


model_path = '/home/xunlong/question_generation/Meta-Llama-3-8B-Instruct-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

question="How many speakers are there in the audio clip, and what are their ethnic groups and genders?"
answer="one Chinese male and one Indian female"

messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question, reference=answer)}]
                      
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, do_sample=True)

response = outputs[0][input_ids[0].shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))


