
GPU=$1
PORT=$2

export CUDA_VISIBLE_DEVICES=$GPU

python -m vllm.entrypoints.openai.api_server \
        --model /scratch/users/astar/ares/zoux/workspaces/models/Meta-Llama-3.1-8B-Instruct \
        --max-model-len 4096 \
        --port $PORT \
        --disable-log-requests \
        --disable-log-stats
        

