
python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/other_prepared/test/NLB_ASR_300_v1
python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/other_prepared/test/NLB_DS_300_v1
python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/other_prepared/test/NLB_SQA_300_v1
python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/other_prepared/train/NLB_ASR_300_v1
python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/other_prepared/train/NLB_DS_300_v1
python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/other_prepared/train/NLB_SQA_300_v1

sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/train

sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/nlb_data/test

sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/check_sample_filter.py /mnt/data/all_datasets/datasets_multimodal

sleep 5

echo "Done"

