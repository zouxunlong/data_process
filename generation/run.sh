
python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_QA.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/common_voice_17_en_ASR_v2"
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_QA.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/common_voice_17_en_ASR_v2/*"
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_QA.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/peoples_speech_ASR_v2"
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_QA.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/peoples_speech_ASR_v2/*"
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_QA.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/gigaspeech_ASR_v2"
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_QA.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/gigaspeech_ASR_v2/*"
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/common_voice_17_en_ASR_v2" Chinese ZH
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/common_voice_17_en_ASR_v2/*"  Chinese ZH
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/peoples_speech_ASR_v2"  Chinese ZH
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/peoples_speech_ASR_v2/*"  Chinese ZH
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/gigaspeech_ASR_v2"  Chinese ZH
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/gigaspeech_ASR_v2/*"  Chinese ZH
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/common_voice_17_en_ASR_v2" Malay MS
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/common_voice_17_en_ASR_v2/*" Malay MS
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/peoples_speech_ASR_v2" Malay MS
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/peoples_speech_ASR_v2/*" Malay MS
sleep 5

python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/t*/ASR/gigaspeech_ASR_v2" Malay MS
# python /mnt/home/zoux/workspaces/xunlong_working_repo/generation/scripts/gen_ST.py "/mnt/home/zoux/datasets_multimodal/other_prepared/ASR/gigaspeech_ASR_v2/*" Malay MS

echo "complete"