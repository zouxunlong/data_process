from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from multiprocess import set_start_method
import torch
from pprint import pprint
from fire import Fire
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_TEMPLATE = """\
    You're given a Question and a Corresponding Answer:

    Question:
    {question}

    Answer:
    {answer}

    [System]
    Please rewrite the question and the answer to be more natural and complete.
    Please strictly make sure they convey exactly the same meaning and sounds nature as a question and answer pair.
    Your response should be formatted as follows:
    Question: (your rephrased question)
    Answer: (your rephrased answer)"""


model_path = '/mnt/home/zoux/workspace/models/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto")


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    directories.sort()
    return directories


def main(dir_path):

    def qa_process(batch, rank):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

        messages = [[{"role": "user", "content": PROMPT_TEMPLATE.format(question=instruction["text"], answer=answer["text"])}]
                    for instruction, answer in
                    zip(batch["instruction"], batch["answer"])]

        input_ids = tokenizer.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.eos_token_id, do_sample=False)

        response_ids = torch.stack(
            [output[input_ids[i].shape[-1]:] for i, output in enumerate(outputs)])
        responses = tokenizer.batch_decode(
            response_ids, skip_special_tokens=True)


        batch["instruction"] = [{"text": response.split("\nAnswer:")[-2].split("Question:")[-1].strip(), "audio": None} for response in responses]
        batch["answer"] = [{"text": response.split("\nAnswer:")[-1].strip(), "audio": None} for response in responses]
        return batch

    for split in get_all_split(dir_path):
        if os.path.exists(split.replace("/Paralingual/", "/Paralingual_rephrased/")):
            continue
        print("start {}".format(split), flush=True)
        dataset = load_from_disk(split)
        dataset = dataset.map(
            qa_process,
            with_rank=True,
            batched=True,
            batch_size=300,
            num_proc=1,
            load_from_cache_file=True,
            writer_batch_size=1,
            desc="rephrase paralinguistic QA",
        )
        dataset.save_to_disk(split.replace("/Paralingual/", "/Paralingual_rephrased/"), num_proc=4)
        print("complete {}".format(split), flush=True)

    print("All done", flush=True)


if __name__ == '__main__':
    # set_start_method("spawn")
    Fire(main)
