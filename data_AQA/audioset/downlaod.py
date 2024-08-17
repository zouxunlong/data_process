from datasets import load_dataset

dataset = load_dataset("agkphysics/AudioSet", "unbalanced", num_proc=16)

dataset.save_to_disk("AudioSet_unbalanced", num_proc=16)


