

python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/dream_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/librispeech_clean_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/librispeech_*_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/ODSQA_zh_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/SLUE_Phase_2_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/Spoken-SQuAD_SQA_v1"
sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/other_prepared/SQA/dream_SQA_v1/*"
sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/other_prepared/SQA/librispeech_SQA_v1/*"
sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/other_prepared/SQA/SLUE_Phase_2_SQA_v1/*"
sleep 5




python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/nlb_data/t*/SQA/NLB_*_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/nlb_data/other_prepared/t*/NLB_300_SQA_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/nlb_data/t*/SQA/NLB_*_DS_v1"
sleep 5
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/gen_fix.py "/mnt/data/all_datasets/nlb_data/other_prepared/t*/NLB_300_DS_v1"
sleep 5