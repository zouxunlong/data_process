
# no_proxy=localhost,127.0.0.1,10.104.0.0/21
# https_proxy=http://10.104.4.124:10104
# http_proxy=http://10.104.4.124:10104
# export HF_ENDPOINT=http://hf-mirror.com
# export HF_ENDPOINT=http://huggingface.co

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface

export CUDA_VISIBLE_DEVICES=0
port=5000

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=1
port=5001

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=2
port=5002

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=3
port=5003

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=4
port=5004

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=5
port=5005

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=6
port=5006

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"



export CUDA_VISIBLE_DEVICES=7
port=5007

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
echo "Started server on port $port"




# bash host_model_judge_llama_3_70b_instruct_awq.sh
