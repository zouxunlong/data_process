
GPU=$1
PORT=$2

export CUDA_VISIBLE_DEVICES=$GPU

python -m vllm.entrypoints.openai.api_server \
        --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
        --max-model-len 40960 \
        --port $PORT \
        --disable-log-requests \
        --disable-log-stats
        

