
import traceback
import os
from datasets import load_from_disk


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def swap_speakers(text):
    """Swaps speaker labels if speaker2 starts the conversation.

    Args:
        text: A string containing the conversation with speaker labels.

    Returns:
        A string with speaker labels potentially swapped.
    """
    lines = text.split('\n')
    if "<Speaker2>:" in lines[0]:
        swapped=True
    else:
        swapped=False

    if swapped:
        lines = [line.replace("<Speaker1>:", "speaker2:").replace("<Speaker2>:", "<Speaker1>:").replace("speaker2:", "<Speaker2>:") for line in lines]
    return swapped, "\n".join(lines)


def map_fn(batch):
    try:
        answer=batch["answer"][0]
        other_attributes=batch["other_attributes"][0].copy()
        swapped, text = swap_speakers(answer["text"])
        if swapped:
            batch["answer"][0]={"audio":None, "text":text}
            batch["other_attributes"][0]["speaker1"]=other_attributes["speaker2"]
            batch["other_attributes"][0]["speaker2"]=other_attributes["speaker1"]
        return batch

    except Exception as e:
        print(traceback.format_exc(), flush=True)


def reformat(split, workers=100):

    print("start {}".format(split), flush=True)
    
    ds = load_from_disk(split)
    ds = ds.map(map_fn,
                batched           = True,
                batch_size        = 1,
                writer_batch_size = 1,
                features          = ds.features,
                num_proc          = workers)
    ds.save_to_disk(split.replace("/IMDA_ASR_", "/swapped/IMDA_ASR_"), num_proc=10)

    print("complete {}".format(split), flush=True)


def main():
    splits_30 = get_all_split("/mnt/home/zoux/datasets/xunlong_working_repo/IMDA_ASR_30_v1")
    splits_60 = get_all_split("/mnt/home/zoux/datasets/xunlong_working_repo/IMDA_ASR_60_v1")
    splits_120 = get_all_split("/mnt/home/zoux/datasets/xunlong_working_repo/IMDA_ASR_120_v1")
    splits = splits_30 + splits_60 + splits_120
    splits.sort()
    for split in splits:
        if os.path.exists(split.replace("/IMDA_ASR_", "/swapped/IMDA_ASR_")):
            continue
        reformat(split)
    print("complete all", flush=True)


if __name__ == "__main__":
    main()
