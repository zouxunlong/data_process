from datasets import load_from_disk


def check(sample):
    return len(sample["context"]["audio"]["array"])>1000


ds = load_from_disk("/mnt/data/all_datasets/ASR_normalized/test/ASR/gigaspeech_ASR_v2")
print(len(ds), flush=True)
print(ds[20003], flush=True)
# ds=ds.select(range(20000, 25619)).filter(lambda x: check(x), batch_size=1, writer_batch_size=1)
# print(len(ds), flush=True)



