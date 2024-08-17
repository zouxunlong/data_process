from collections import defaultdict
from datasets import load_from_disk, Audio, Features, Value
import soundfile as sf

def main(ds_path, workers=8):

    def reformat(batch):
        new_batch = {
            "context": [],
            "instruction": [],
            "answer": [],
            "other_attributes": [],
        }
        for i, context in enumerate(batch["context"]):
            if context.strip():
                continue

            new_batch["context"].append({
                "text": "",
                "audio": None
            })
            new_batch["instruction"].append({
                "text": batch["input"][i],
                "audio": batch["audio"][i]
            })
            new_batch["answer"].append({
                "text": batch["output"][i],
                "audio": None
            })
            new_batch["other_attributes"].append({
                "index": batch["index"][i],
            })
        return new_batch

    print("start {}". format(ds_path), flush=True)
    ds = load_from_disk(ds_path)
    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, mono=True, decode=True, id=None)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, mono=True, decode=True, id=None)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, mono=True, decode=True, id=None)},
        'other_attributes': {
            "index": ds.features["index"],
        }
    })
    ds = ds.map(
        reformat,
        batched=True,
        batch_size=1,
        features=features,
        remove_columns=ds.column_names,
        num_proc=workers,
        writer_batch_size=1
    )
    ds_dict=ds.train_test_split(test_size=2000)
    ds_dict.save_to_disk("alpaca-gpt4-audio.hf")
    print("complete {}". format(ds_path), flush=True)


def sample(ds_path):
    ds=load_from_disk(ds_path)
    print(ds[2])
    sf.write("sample.wav", data=ds[2]["context"]["audio"]["array"], samplerate=16000)


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
        batch_size=1,
        writer_batch_size=1,
        features=ds["test"].features,
        num_proc=workers,
    )
    ds.save_to_disk("alpaca-gpt4-audio_v1", num_proc=4)
    print("complete {}". format(ds_path), flush=True)

if __name__ == "__main__":
    # sample("/home/all_datasets/pre_ready_datasets/xunlong_working_repo/alpaca-gpt4-audio_v1/train")
    sample("/home/all_datasets/datasets_multimodal/SI/alpaca-gpt4-audio_v1/test")
    # reformat("/home/all_datasets/datasets_multimodal/SI/alpaca-gpt4-audio_v1")
