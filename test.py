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

dirs=os.listdir("/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/train/ASR")
for dir in dirs:
    # ds=load_from_disk(f"/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/train/ASR/{dir}").select(range(5))
    ds=load_from_disk(f"/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/train/ASR/common_voice_17_en_ASR_v3").select(range(5))
    for item in ds:
        print(f"================{dir}=================", flush=True)
        print(item["instruction"]["text"], flush=True)
        print(item["answer"]["text"], flush=True)
    print("===============================================================", flush=True)
    print("===============================================================", flush=True)
    break

