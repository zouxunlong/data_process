from datasets import load_from_disk
from tqdm import tqdm
import re
from multiprocessing import Pool
from fire import Fire
import logging
import string
import unicodedata


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


translator = str.maketrans('', '', string.punctuation)

def normalize_sentence(sentence):
    sentence = unicodedata.normalize('NFKC', sentence.translate(translator))
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
    sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence




def map_fn(batch, id_script_map):

    id=batch["other_attributes"][0]["id"]
    transcription=id_script_map.get(id, None)
    assert transcription, "transcription not found for id: {}".format(id)
    batch["answer"][0]["text"]=transcription

    return batch



def main(workers=20):

    for part in ["PART1"]:

        test_path  = f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/ASR/IMDA_{part}_ASR_v4"
        train_path = f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/train/ASR/IMDA_{part}_ASR_v4"
        
        
        ds_test  = load_from_disk(test_path)
        ds_train = load_from_disk(train_path)

        print(ds_train, flush=True)
        print(ds_test, flush=True)

        test_transcriptions=set()
        for sample in tqdm(ds_test["answer"]):
            test_transcriptions.add(normalize_sentence(sample["text"]).lower().strip())

        breakpoint()
        print(len(test_transcriptions), flush=True)
        print(len(ds_train), flush=True)

        ds_train = ds_train.filter(lambda x: [answer["text"].replace("<Speaker1>: ", "").lower().strip() not in test_transcriptions for answer in x["answer"]], 
                                batched=True, batch_size=1000, writer_batch_size=1000, num_proc=workers)
        print(len(ds_train), flush=True)

        ds_train.save_to_disk(train_path.replace("ASR_v4", "ASR_v5"), num_proc=10)



if __name__ == "__main__":
    Fire(main)
