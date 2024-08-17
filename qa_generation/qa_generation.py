from transformers import pipeline
import torch
import os
import fire

model_path = '/home/xunlong/question_generation/Meta-Llama-3-8B-Instruct-hf'


def main():

    pipe = pipeline("text-generation", model=model_path,
                    torch_dtype=torch.bfloat16, device_map='auto')

    questions = [
        {
            "role": "system",
            "content": "You are given a caption of an audio recording. Please responds with one reading comprehension question that can be answered using only the information contained in the context. Do not include the answer.",
        },
        {
            "role": "user",
            "content": "I will arrive home by 10pm."
        },
    ]

    # Applies chat template to messages and tokenize text when neccesary for model
    prompt = pipe.tokenizer.apply_chat_template(
        questions,
        tokenize=False,
        add_generation_prompt=True)

    # Generates questions
    with torch.no_grad():
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True,
                       temperature=0.7, top_k=50, top_p=0.95)

    # Extracts and prints the system-generated question
    generated_question = outputs[0]["generated_text"].split("<|assistant|>")[
        1].strip()

    generated_question = generated_question.split('\n')[0].strip()

    questions = [
        {
            "role": "system",
            "content": "Please answer the question based on the audio contenxt. Be concise.",
        },
        {
            "role": "user",
            "content": 'Audio Context: ' + "I will arrive home by 10pm." + 'Quesiton: ' + generated_question},]

    # Applies chat template to messages and tokenize text when neccesary for model
    prompt = pipe.tokenizer.apply_chat_template(
        questions, tokenize=False, add_generation_prompt=True)

    # Generates questions
    with torch.no_grad():
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # Extracts and prints the system-generated question
    generated_answer = outputs[0]["generated_text"].split("<|assistant|>")[1].strip()

    print(generated_question, flush=True)
    print(generated_answer, flush=True)


if __name__ == '__main__':
    fire.Fire(main)
