from datasets import load_from_disk
import soundfile as sf
import json


ds=load_from_disk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/ASR/IMDA_PART1_ASR_v4")
for i, item in enumerate(ds):
    sf.write(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_sample/AR/{i}.wav", item["context"]["audio"]["array"], 16000)
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_sample/AR/{i}.txt", "w").write(json.dumps(item["other_attributes"]["speaker"], ensure_ascii=False, indent=4))

