from datasets import load_from_disk

# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/Fhios_Data_en_30_ASR")

# for i, item in enumerate(ds):
#     print(i, flush=True)

# ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/tamil_speech_data_4700024041and4700024043_ta_30_ASR")

# for i, item in enumerate(ds):
#     print(i, flush=True)

ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/tamil_speech_data_4700024041and4700024043_ta_300_ASR")

for i, item in enumerate(ds):
    print(i, flush=True)

ds=load_from_disk("/mnt/data/all_datasets/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/burmese_speech_dataset_my_30_ASR")

for i, item in enumerate(ds):
    print(i, flush=True)

