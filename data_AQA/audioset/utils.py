
from datasets import load_from_disk


def filter_sppech():

    def speech_filter(batch):
        return [True if 'Speech' in human_labels else False for human_labels in batch["human_labels"]]

    dd=load_from_disk("AudioSet_unbalanced")

    for split, ds in dd.items():
        ds = ds.filter(speech_filter, batched=True, num_proc=16)
        ds.save_to_disk("{}/{}".format("AudioSet_unbalanced_speech", split), num_proc=16)
        print("completed {}".format(split), flush=True)

filter_sppech()
