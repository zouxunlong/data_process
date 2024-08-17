from datasets import load_from_disk


def examine_sample(path):
    ds=load_from_disk(path)
    print(ds[1], flush=True)
    print(ds.column_names, flush=True)
    print("samples:{}".format(len(ds)), flush=True)

if __name__=="__main__":
    path="/home/all_datasets/multimodal_datasets/ASR/common_voice_15_en_shuffle.schemed/test"
    examine_sample(path)
