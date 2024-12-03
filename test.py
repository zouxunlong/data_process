from datasets import load_from_disk
import os

# root="/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_nlb/train/ASR"
# dirs=os.listdir(f"{root}")
# print(dirs)
# for dir in dirs:
#     ds = load_from_disk(f"{root}/{dir}")
#     print("=================================", flush=True)
#     print(f"{dir}:", flush=True)
#     print(f"{ds[0]['answer']['text']}", flush=True)
#     instructions=set()
#     for i, item in enumerate(ds):
#         instructions.add(item["instruction"]["text"])
#         if i==20:
#             break
#     print(f"{dir}: {len(instructions)}", flush=True)

dirs=os.listdir("/mnt/data/all_datasets/backup/ASR_new")
for dir in dirs:
    os.rename



# IMDA_PART4_ASR_v4
# IMDA_PART5_ASR_v4
# IMDA_PART3_ASR_v4
# IMDA_PART6_ASR_v4

