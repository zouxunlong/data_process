from datasets import load_from_disk, Features, Value, Audio



ds=load_from_disk('/mnt/home/zoux/datasets/NLB/test/NLB_v1')

print(ds.features, flush=True)
