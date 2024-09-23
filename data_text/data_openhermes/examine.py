from datasets import load_from_disk

dataset = load_from_disk("/home/user/data/data_text/data_openhermes/openhermes2_5.hf")

print(dataset[0], flush=True)
