for i in $(seq 1 6000);
do
    echo $i
    wget https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk10/example_train_$i.jsonl.zst
done