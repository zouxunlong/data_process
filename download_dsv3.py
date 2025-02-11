


# # Use a pipeline as a high-level helper
# from transformers import pipeline
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="deepseek-ai/DeepSeek-V3")


# # Load model directly
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3", trust_remote_code=True)
pipe(messages)