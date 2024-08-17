from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = '/home/xunlong/question_generation/Meta-Llama-3-8B-Instruct-hf'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "Who are you?"},]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, do_sample=False)

response = outputs[0][input_ids[0].shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
