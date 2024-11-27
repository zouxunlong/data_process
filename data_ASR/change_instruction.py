from datasets import load_from_disk
import random


instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


ds=load_from_disk("/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/train/ASR/gigaspeech_ASR_v2")
# ds=ds.map(lambda x: {"instruction": {"text": random.choice(instructions_asr), "audio": None}}, num_proc=112)
print(ds[0], flush=True)
# ds.save_to_disk("/scratch/users/astar/ares/zoux/datasets/datasets_hf_bytes/datasets_multimodal/train/ASR/gigaspeech_ASR_v3", num_proc=8)
