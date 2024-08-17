from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch
from fire import Fire
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROMPT_TEMPLATE = """\
    [Question]
    {question}

    [Ground Truth Reference]
    {reference}

    [System]
    You're given the question and the groundtruth reference based on an audio clip.
    Please rewrite the groundtruth reference as one complete sentence.
    Please strictly make sure they convey the same meaning and sounds nature as a response to the question based on the audio clip.
    Your response should be formatted as follows:
    Explanation: (Provide a brief explanation.)
    Answer: (your rephrased answer)"""


model_path = '/home/xunlong/question_generation/Meta-Llama-3-8B-Instruct-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")


def main():

    for split in ["validation"]:

        def qa_process(batch):

            messages=[[{"role": "user", "content": PROMPT_TEMPLATE.format(question=instruction["text"], reference=answer["text"])}]
                      for instruction, answer in 
                      zip(batch["instruction"], batch["answer"])]

            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=True,
                add_generation_prompt=True,
                return_tensors="pt").to(model.device)

            outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
            response_ids = torch.stack([output[input_ids[i].shape[-1]:] for i, output in enumerate(outputs)])
            with torch.no_grad():
                responses=tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            batch["answer"]=[{"audio":None, "text":response.split("\nAnswer:")[-1].strip()} for response in responses]
            return batch

        dataset = load_from_disk('/data/xunlong/clotho_aqa.schemed/{}'.format(split))
        dataset = dataset.map(
            qa_process,
            batched=True,
            batch_size=80,
            num_proc=1,
            load_from_cache_file=True,
            writer_batch_size=35,
            keep_in_memory=False,
            desc="Answer rephrase",
        )

        dataset.save_to_disk('/data/xunlong/clotho_aqa.rephrase.schemed/{}'.format(split), num_proc=4)


if __name__ == '__main__':
    Fire(main)


