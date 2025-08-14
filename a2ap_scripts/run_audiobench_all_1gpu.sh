#!/bin/bash
#PBS -N AudioBench
#PBS -l select=1:ncpus=14:ngpus=1:mem=235gb:container_engine=enroot
#PBS -l walltime=120:00:00
#PBS -q normal
#PBS -P 13003558
#PBS -l container_image=/scratch/users/astar/ares/suns1/containers/multimodal_trainer_pytorch_2.4.sqsh
#PBS -l container_name=multimodal_trainer
#PBS -l enroot_env_file=/home/users/astar/ares/wonghmj/multimodal_trainer/scripts/a2ap_scripts/enroot_scripts/env.conf
#PBS -o /home/users/astar/ares/wonghmj/logs/stdout.txt
#PBS -e /home/users/astar/ares/wonghmj/logs/stderr.txt

ROOT_DIR=/home/users/astar/ares/wonghmj
PROJ_DIR=${ROOT_DIR}/multimodal_trainer

MODEL_DIR=${PROJ_DIR}/exp/stage2_only_adaptoronly_emotion2vecpluslarge_fft_mlp100_gemma2_9b_lora/checkpointbak-120000
OUT_DIR=${MODEL_DIR}/audiobench

AUDIOBENCH_REPO_DIR=/data/projects/13003558/audiobench

HF_HOME=/data/projects/13003558/hf_cache

JUDGE=gpt4o_judge
MODEL_TYPE=MERaLiON_AudioLLM_v1
DATASETS="slue_p2_sqa5_test public_sg_speech_qa_test spoken_squad_test openhermes_audio_test alpaca_audio_test clotho_aqa_test wavcaps_qa_test audiocaps_qa_test wavcaps_test audiocaps_test voxceleb_accent_test imda_30s_sqa_test imda_30s_sqa_human_test imda_part3_30s_sqa_test imda_part3_30s_sqa_human_test imda_part4_30s_sqa_test imda_part4_30s_sqa_human_test imda_part5_30s_sqa_test imda_part5_30s_sqa_human_test imda_part6_30s_sqa_test imda_part6_30s_sqa_human_test imda_30s_ds_test imda_30s_ds_human_test imda_part3_30s_ds_test imda_part3_30s_ds_human_test imda_part4_30s_ds_test imda_part4_30s_ds_human_test imda_part5_30s_ds_test imda_part5_30s_ds_human_test imda_part6_30s_ds_test imda_part6_30s_ds_human_test"
DATASETS_BINARY="cn_college_listen_mcq_test dream_tts_mcq_test iemocap_emotion_test meld_sentiment_test meld_emotion_test voxceleb_gender_test iemocap_gender_test mu_chomusic_test imda_30s_gr_test"
DATASETS_METEOR="wavcaps_test audiocaps_test"
DATASETS_WER="librispeech_test_clean librispeech_test_other common_voice_15_en_test peoples_speech_test gigaspeech_test tedlium3_test tedlium3_long_form_test earnings21_test earnings22_test aishell_asr_zh_test imda_part1_asr_test imda_part2_asr_test imda_part3_30s_asr_test imda_part4_30s_asr_test imda_part5_30s_asr_test imda_part6_30s_asr_test cna_test idpc_test parliament_test ukusnews_test mediacorp_test"
DATASETS_BLEU="covost2_en_id_test covost2_en_zh_test covost2_en_ta_test covost2_id_en_test covost2_zh_en_test covost2_ta_en_test"

pbs_env=`env | grep -i "PBS_" | cut -d "=" -f 1 | sed '{:q;N;s/\n/ -e /g;t q}'`

enroot start \
	-m /data:/data -m /scratch:/scratch \
  -e $pbs_env \
	-e HF_HOME=$HF_HOME \
	multimodal_trainer \
	bash -c "
	cd $PROJ_DIR
  mkdir -p $OUT_DIR
  for d in $DATASETS; do
    python ${AUDIOBENCH_REPO_DIR}/src/main_evaluate.py --dataset_name \$d --model_name $MODEL_TYPE --batch_size 1 --overwrite False --metrics $JUDGE --number_of_samples -1 --multimodal_trainer_path $PROJ_DIR --checkpoint $MODEL_DIR --out_dir $OUT_DIR 2> ${OUT_DIR}/\${d}.stderr.txt 1> ${OUT_DIR}/\${d}.stdout.txt
  done
  for d in $DATASETS_BINARY; do
    python ${AUDIOBENCH_REPO_DIR}/src/main_evaluate.py --dataset_name \$d --model_name $MODEL_TYPE --batch_size 1 --overwrite False --metrics ${JUDGE}_binary --number_of_samples -1 --multimodal_trainer_path $PROJ_DIR --checkpoint $MODEL_DIR --out_dir $OUT_DIR 2> ${OUT_DIR}/\${d}.stderr.txt 1> ${OUT_DIR}/\${d}.stdout.txt
  done
  for d in $DATASETS_METEOR; do
    python ${AUDIOBENCH_REPO_DIR}/src/main_evaluate.py --dataset_name \$d --model_name $MODEL_TYPE --batch_size 1 --overwrite False --metrics meteor --number_of_samples -1 --multimodal_trainer_path $PROJ_DIR --checkpoint $MODEL_DIR --out_dir $OUT_DIR 2> ${OUT_DIR}/\${d}.stderr.txt 1> ${OUT_DIR}/\${d}.stdout.txt
  done
  for d in $DATASETS_WER; do
    python ${AUDIOBENCH_REPO_DIR}/src/main_evaluate.py --dataset_name \$d --model_name $MODEL_TYPE --batch_size 1 --overwrite False --metrics wer --number_of_samples -1 --multimodal_trainer_path $PROJ_DIR --checkpoint $MODEL_DIR --out_dir $OUT_DIR 2> ${OUT_DIR}/\${d}.stderr.txt 1> ${OUT_DIR}/\${d}.stdout.txt
  done
  for d in $DATASETS_BLEU; do
    python ${AUDIOBENCH_REPO_DIR}/src/main_evaluate.py --dataset_name \$d --model_name $MODEL_TYPE --batch_size 1 --overwrite False --metrics bleu --number_of_samples -1 --multimodal_trainer_path $PROJ_DIR --checkpoint $MODEL_DIR --out_dir $OUT_DIR 2> ${OUT_DIR}/\${d}.stderr.txt 1> ${OUT_DIR}/\${d}.stdout.txt
  done
  python scripts/jeremy_scripts/collate_audiobench_scores.py --audiobench-dir ${OUT_DIR}/${MODEL_TYPE} --out-dir ${OUT_DIR}/${MODEL_TYPE}
  "
