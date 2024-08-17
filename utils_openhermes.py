from collections import defaultdict
from datasets import load_from_disk, Audio, Features, Value
import soundfile as sf


def sample(ds_path):
    ds=load_from_disk(ds_path).select([0])
    print(ds[0])
    sf.write("sample.wav", data=ds[0]["instruction"]["audio"]["array"], samplerate=16000)


def reformat(ds_path, workers=16):

    def reformat(batch):
        new_batch=defaultdict(list)
        for i in range(len(batch["context"])):
            new_batch["context"].append({
                "text": "",
                "audio": batch["instruction"][i]["audio"]
            })
            new_batch["instruction"].append({
                "text": "Please follow the instruction in the speech",
                "audio": None
            })
        return new_batch

    print("start {}". format(ds_path), flush=True)
    ds = load_from_disk(ds_path)
    ds = ds.map(
        reformat,
        batched=True,
        batch_size=20,
        writer_batch_size=1,
        features=ds["test"].features,
        num_proc=workers,
    )
    ds.save_to_disk("openhermes-audio_v1", num_proc=4)
    print("complete {}". format(ds_path), flush=True)


if __name__ == "__main__":
    # sample("/home/all_datasets/datasets_multimodal/SI/openhermes-audio_v1/test")
    reformat("/home/all_datasets/datasets_multimodal/SI/openhermes-audio_v1")
