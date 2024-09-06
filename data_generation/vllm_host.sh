
GPU=$1
PORT=$2

export CUDA_VISIBLE_DEVICES=$GPU

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --port $PORT \
        --disable-log-requests \
        --disable-log-stats
        

