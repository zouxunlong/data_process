
python /mnt/data/all_datasets/xunlong_working_repo/check_loading.py /mnt/data/all_datasets/nlb_data
echo "check loading for nlb_data completed"
sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/check_loading.py /mnt/data/all_datasets/datasets_multimodal
echo "check loading for multimodal completed"

sleep 5

python /mnt/data/all_datasets/xunlong_working_repo/check_loading.py /mnt/data/all_datasets/datasets_text

echo "check loading for datasets_text completed"

