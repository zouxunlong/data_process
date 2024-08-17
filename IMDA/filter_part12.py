import json
import traceback
from datasets import load_from_disk


conversation_id_dict=json.load(open("/home/user/data/data_IMDA/res.conversation_id.json"))


def filtering(example, conversation_ids):
    try:
        conversation_id=example["other_attributes"]["conversation_id"][1:]
        if conversation_id in conversation_ids:
            return False
        else:
            return True
    except:
        print(traceback.format_exc(), flush=True)
        return False


def main(*args):
    for part in args:
        ds=load_from_disk("/home/all_datasets/datasets_multimodal/ASR/IMDA.ASR.schemed/{}.ASR.schemed/train".format(part))
        print(ds.column_names, flush=True)
        ds=ds.filter(filtering, fn_kwargs={"conversation_ids": conversation_id_dict[part]["conversation_id"]}, num_proc=8, batch_size=1000, writer_batch_size=1000)
        ds.save_to_disk("/home/all_datasets/datasets_multimodal/ASR/IMDA.ASR.new.schemed/{}.ASR.schemed/train".format(part), num_proc=4)


if __name__=="__main__":
    from fire import Fire
    Fire(main)
