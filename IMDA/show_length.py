from datasets import load_from_disk


ds=load_from_disk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/test/ASR/IMDA_PART3_ASR_v4")
for i, item in enumerate(ds):
    print(item["other_attributes"]["end"]-item["other_attributes"]["start"], flush=True)
