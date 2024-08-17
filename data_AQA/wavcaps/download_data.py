from datasets import load_from_disk
dataset = load_from_disk("/home/user/data/data_AQA/wavcaps/WavCaps_hf/validation")
for i, item in enumerate(dataset):
    print(item["audio_path"], flush=True)
    print(i, flush=True)
