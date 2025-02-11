# Load model directly
from transformers import AutoModelForCausalLM, QuantoConfig

quantization_config = QuantoConfig(weights="int8")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True, quantization_config=quantization_config)

print(model)
