

python /mnt/home/zoux/xunlong_working_repo/qa_ds_generation/rephrase_paralingual.py "/mnt/home/zoux/datasets/IMDA/IMDA_GR_*_v1/*/*"

sleep 15

python /mnt/home/zoux/xunlong_working_repo/qa_ds_generation/rephrase_paralingual.py "/mnt/home/zoux/datasets/IMDA/IMDA_NR_*_v1/*/*"

sleep 15

python /mnt/home/zoux/xunlong_working_repo/qa_ds_generation/rephrase_paralingual.py "/mnt/home/zoux/datasets/IMDA/IMDA_MIX_*_v1/*/*"

echo "complete"