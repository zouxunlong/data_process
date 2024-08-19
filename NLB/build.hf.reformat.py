
from datasets import load_from_disk, Audio, Features, Value


def map_fn(example):

    return {"context": {"audio": example['audio'], "text": None}}


def build_asr_ds(workers=30):

    ds = load_from_disk(
        "/mnt/data/all_datasets/nlb_data/other_prepared/train/NLB_v1")

    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'transcriptions': ds.features['transcriptions'],
    })

    ds = ds.map(
        map_fn,
        features=features,
        num_proc=workers,
        batch_size=1,
        writer_batch_size=1,
        remove_columns="audio",
    )
    print(ds, flush=True)
    ds.save_to_disk("/mnt/data/all_datasets/nlb_data/other_prepared/train/NLB_v1_v1", num_proc=4)


if __name__ == "__main__":
    import fire
    fire.Fire(build_asr_ds)
