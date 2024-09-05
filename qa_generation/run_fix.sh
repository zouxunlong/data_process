
python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/DS_gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/t*/SQA/IMDA_PART*_DS_v1"

sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/QA_gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/other_prepared/SQA/IMDA_300_SQA/IMDA_PART*_300_SQA_v1/t*"

sleep 5


python /mnt/data/all_datasets/xunlong_working_repo/qa_generation/DS_gen_fix.py "/mnt/data/all_datasets/datasets_multimodal/other_prepared/SQA/IMDA_300_DS/IMDA_PART*_300_DS_v1/t*"

echo "done"
