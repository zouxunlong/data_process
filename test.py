from datasets import load_from_disk, Value, Audio, ClassLabel, Features


ds = load_from_disk("/mnt/data/all_datasets/datasets_multimodal_opus_bytes/train/ASR/gigaspeech_ASR_v2")

features=ds.features
features['context']['audio'] = Audio(sampling_rate=16000, mono=True, decode=False, id=None)

ds = ds.cast(
    features=features,
    num_proc=16)


print(ds)
print(ds.features)
