ROOT_DIR=/home/users/astar/ares/wonghmj
PROJ_DIR=${ROOT_DIR}/multimodal_trainer

MODEL_DIR=${PROJ_DIR}/exp/stage2_only_adaptoronly_emotion2vecpluslarge_fft_mlp100_gemma2_9b_lora_nolayerdrop/checkpoint-120000
OUT_DIR=${MODEL_DIR}/audiobench

JUDGE=gpt4o_judge
MODEL_TYPE=MERaLiON_AudioLLM_v1

DATASETS="slue_p2_sqa5_test public_sg_speech_qa_test spoken_squad_test openhermes_audio_test alpaca_audio_test clotho_aqa_test wavcaps_qa_test audiocaps_qa_test wavcaps_test audiocaps_test voxceleb_accent_test imda_30s_sqa_test imda_30s_sqa_human_test imda_part3_30s_sqa_test imda_part3_30s_sqa_human_test imda_part4_30s_sqa_test imda_part4_30s_sqa_human_test imda_part5_30s_sqa_test imda_part5_30s_sqa_human_test imda_part6_30s_sqa_test imda_part6_30s_sqa_human_test imda_30s_ds_test imda_30s_ds_human_test imda_part3_30s_ds_test imda_part3_30s_ds_human_test imda_part4_30s_ds_test imda_part4_30s_ds_human_test imda_part5_30s_ds_test imda_part5_30s_ds_human_test imda_part6_30s_ds_test imda_part6_30s_ds_human_test"
DATASETS_BINARY="cn_college_listen_mcq_test dream_tts_mcq_test iemocap_emotion_test meld_sentiment_test meld_emotion_test voxceleb_gender_test iemocap_gender_test mu_chomusic_test imda_30s_gr_test"
DATASETS_METEOR="wavcaps_test audiocaps_test"
DATASETS_WER="librispeech_test_clean librispeech_test_other common_voice_15_en_test peoples_speech_test gigaspeech_test tedlium3_test tedlium3_long_form_test earnings21_test earnings22_test aishell_asr_zh_test imda_part1_asr_test imda_part2_asr_test imda_part3_30s_asr_test imda_part4_30s_asr_test imda_part5_30s_asr_test imda_part6_30s_asr_test cna_test idpc_test parliament_test ukusnews_test mediacorp_test"
DATASETS_BLEU="covost2_en_id_test covost2_en_zh_test covost2_en_ta_test covost2_id_en_test covost2_zh_en_test covost2_ta_en_test"

for d in $DATASETS; do
  qsub -v DATASET=${d},METRIC=${JUDGE},PROJ_DIR=${PROJ_DIR},MODEL_DIR=${MODEL_DIR},OUT_DIR=${OUT_DIR},MODEL_TYPE=${MODEL_TYPE} ${PROJ_DIR}/scripts/a2ap_scripts/run_audiobench.sh
done
for d in $DATASETS_BINARY; do
  qsub -v DATASET=${d},METRIC=${JUDGE}_binary,PROJ_DIR=${PROJ_DIR},MODEL_DIR=${MODEL_DIR},OUT_DIR=${OUT_DIR},MODEL_TYPE=${MODEL_TYPE} ${PROJ_DIR}/scripts/a2ap_scripts/run_audiobench.sh
done
for d in $DATASETS_METEOR; do
  qsub -v DATASET=${d},METRIC=meteor,PROJ_DIR=${PROJ_DIR},MODEL_DIR=${MODEL_DIR},OUT_DIR=${OUT_DIR},MODEL_TYPE=${MODEL_TYPE} ${PROJ_DIR}/scripts/a2ap_scripts/run_audiobench.sh
done
for d in $DATASETS_WER; do
  qsub -v DATASET=${d},METRIC=wer,PROJ_DIR=${PROJ_DIR},MODEL_DIR=${MODEL_DIR},OUT_DIR=${OUT_DIR},MODEL_TYPE=${MODEL_TYPE} ${PROJ_DIR}/scripts/a2ap_scripts/run_audiobench.sh
done
for d in $DATASETS_BLEU; do
  qsub -v DATASET=${d},METRIC=bleu,PROJ_DIR=${PROJ_DIR},MODEL_DIR=${MODEL_DIR},OUT_DIR=${OUT_DIR},MODEL_TYPE=${MODEL_TYPE} ${PROJ_DIR}/scripts/a2ap_scripts/run_audiobench.sh
done
  