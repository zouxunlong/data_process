

python /mnt/home/zoux/workspaces/xunlong_working_repo/qa_generation/QA_gen.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/common_voice_17_en_ASR_v1"
python /mnt/home/zoux/workspaces/xunlong_working_repo/qa_generation/QA_gen.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/common_voice_17_en_ASR_v1/*"

sleep 15

python /mnt/home/zoux/workspaces/xunlong_working_repo/qa_generation/QA_gen.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/gigaspeech_ASR_v1"
python /mnt/home/zoux/workspaces/xunlong_working_repo/qa_generation/QA_gen.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/gigaspeech_ASR_v1/*"

sleep 15

python /mnt/home/zoux/workspaces/xunlong_working_repo/qa_generation/QA_gen.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/peoples_speech_ASR_v11"
python /mnt/home/zoux/workspaces/xunlong_working_repo/qa_generation/QA_gen.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/peoples_speech_ASR_v1/*"

echo "complete"