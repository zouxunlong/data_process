from datasets import load_from_disk
import fire


def filter_fn(x, test_samples):
    conversation_id = x["other_attributes"]["conversation_id"]
    setting         = x["other_attributes"]["setting"]
    start           = x["other_attributes"]["start"]
    end             = x["other_attributes"]["end"]
    for item in test_samples:
        if conversation_id == item[0] and setting.split("-")[0]==item[1] and start <= item[3] and item[2] <= end:
            return False
    return True


def fix(part, chunk_limit):

    split      = f"/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_{chunk_limit}_ASR_v4"
    split_test = f"/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_asr/test/ASR/IMDA_{part}_{chunk_limit}_ASR_v4"

    ds_test = load_from_disk(split_test)

    test_samples=set()
    for sample in ds_test:
        test_samples.add((sample["other_attributes"]["conversation_id"], sample["other_attributes"]["setting"], sample["other_attributes"]["start"], sample["other_attributes"]["end"]))
    print("settings: ", set([item[1] for item in test_samples]), flush=True)

    ds       = load_from_disk(split)
    ds_train = ds.filter(filter_fn, fn_kwargs = {"test_samples": test_samples}, batch_size=1, writer_batch_size=1, num_proc=56)
    ds_train.save_to_disk(f"/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_{chunk_limit}_ASR_v5", num_proc=10)


def train_test_split(part, chunk_limit):
    
    split      = f"/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_{chunk_limit}_ASR_v4"

    ds       = load_from_disk(split)
    ds_train = ds.train_test_split(test_size=1000, seed=42)
    ds_train["train"].save_to_disk(f"/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_{chunk_limit}_ASR_v5", num_proc=10)
    ds_train["test"].save_to_disk(f"/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_asr/test/ASR/IMDA_{part}_{chunk_limit}_ASR_v5", num_proc=10)



def main(chunk_limit):

    if chunk_limit in [30,60,120]:
        for part in ["PART3", "PART4", "PART5", "PART6"]:
            print("start {} {}".format(part, chunk_limit), flush=True)
            fix(part, chunk_limit)

    if chunk_limit==300:
        for part in ["PART3", "PART4", "PART5", "PART6"]:
            print("start {} {}".format(part, chunk_limit), flush=True)
            train_test_split(part, chunk_limit)


if __name__ == "__main__":
    fire.Fire(main)
