

export HF_HOME=~/.cache/huggingface

export CUDA_VISIBLE_DEVICES=2,3

python -m vllm.entrypoints.openai.api_server \
        --model /mnt/home/zoux/models/Meta-Llama-3.1-8B-Instruct \
        --port 8000 \
        --disable-log-requests \
        --disable-log-stats &
        

