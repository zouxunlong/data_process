from datasets import load_dataset


for lang in ["en"]:
    ds = load_dataset("mozilla-foundation/common_voice_17_0", lang, 
                    cache_dir="/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/.cache",
                    token=True,
                    trust_remote_code=True, 
                    num_proc=16)
    ds.save_to_disk("/mnt/data/all_datasets/pre_ready_datasets/xunlong_working_repo/data_ASR/common_voice/common_voice_{}".format(lang), 
                    num_proc=4)
