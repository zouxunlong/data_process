

python /mnt/home/zoux/workspace/IMDA_scripts/qa_ds_generation/QA_gen.py "/mnt/home/zoux/datasets/IMDA/IMDA_ASR_*_v1/*/*"

sleep 5

python /mnt/home/zoux/workspace/IMDA_scripts/qa_ds_generation/DS_gen.py "/mnt/home/zoux/datasets/IMDA/IMDA_ASR_*_v1/*/*"

sleep 5

python /mnt/home/zoux/workspace/IMDA_scripts/qa_ds_generation/QA_gen.py "/mnt/home/zoux/datasets/NLB/*/NLB_ASR_*_v1"

sleep 5

python /mnt/home/zoux/workspace/IMDA_scripts/qa_ds_generation/DS_gen.py "/mnt/home/zoux/datasets/NLB/*/NLB_ASR_*_v1"

echo "complete"