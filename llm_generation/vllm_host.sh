
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# model=casperhansen/llama-3-70b-instruct-awq
# model=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
model=deepseek-ai/DeepSeek-V3


python -m vllm.entrypoints.openai.api_server \
        --model ${model} \
        --port 5000 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 2 \
        --trust-remote-code \
        --disable-log-requests \
        --disable-log-stats

# echo "Started server on port 5000"
