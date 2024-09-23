from datasets import load_dataset

dataset = load_dataset("SeaEval/CRAFT-Singapore-GPT4")

dataset.save_to_disk("/home/user/data/data_text/data_seaeval/CRAFT-Singapore-GPT4.hf")
