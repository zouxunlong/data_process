from glob import glob
from datasets import load_from_disk
from opencc import OpenCC


cc = OpenCC('t2s')


def map_fn(batch):
    answers         = batch["answer"]
    new_answers     = [{"text": cc.convert(answer["text"]), "audio":None,} for answer in answers]
    batch["answer"] = new_answers

    
    for i, other_attribute in enumerate(batch["other_attributes"]):
        batch["other_attributes"][i]["chinese"]=cc.convert(other_attribute["chinese"])

    return batch


def convert(split):
    ds = load_from_disk(split)
    ds = ds.map(map_fn, batched=True, num_proc=28)
    ds.save_to_disk(split+"_new", num_proc=1)


def main():
    splits=glob("/data/projects/13003558/zoux/workspace/data_process/_data_in_processing/taiwanese_sentences_hok_30_ASR")
    for split in splits:
        print(f"start {split}")
        convert(split)


if __name__ == "__main__":
    main()

