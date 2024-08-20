

from datasets import load_dataset
dataset = load_dataset("jp1924/AudioCaps")
dataset.save_to_disk("jp1924_AudioCaps")
print("complete", flush=True)

