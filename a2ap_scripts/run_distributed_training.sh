#!/bin/bash
#PBS -N training
#PBS -l select=1:ncpus=112:ngpus=8:mem=1887gb:container_engine=enroot
#PBS -l walltime=120:00:00
#PBS -q normal
#PBS -P 13003558
#PBS -j oe
#PBS -l container_image=/scratch/users/astar/ares/suns1/containers/multimodal_trainer_pytorch_2.4.sqsh
#PBS -l container_name=multimodal_trainer
#PBS -l enroot_env_file=/scratch/users/astar/ares/suns1/workspace/multimodal_trainer/scripts/a2ap_scripts/enroot_scripts/env.conf

ROOT_DIR=/scratch/users/astar/ares/suns1
PROJ_DIR=$ROOT_DIR/workspace/multimodal_trainer

#EXP_NAME=debug
#EXP_NAME=stage2_whisper3_fft_mlp100_gemma2_9b_lora
#EXP_NAME=stage1_whisper3_clap_fft_mlp100_llama3.1_8b_instruct_lora
#EXP_NAME=stage1_whisper3_fft_mlp100_llama3.2_3b_instruct_lora
#EXP_NAME=stage1_whisper3_fft_mlp100_llama3.2_1b_instruct_lora
#EXP_NAME=stage0_whisper3_fft_mlp100_gemma2_9b_frozens
#EXP_NAME=stage0_whisper3_fft_mlp100_llama3.1_8b_frozen

#EXP_NAME=stage0_whisper3_fft_mlp100_sealion2.1_8b_frozen
#EXP_NAME=stage0_whisper3_fft_mlp100_gemma2_9b_freeze
#EXP_NAME=stage0.5_whisper3_fft_mlp100_gemma2_9b_lora
#EXP_NAME=stage0.5_whisper3_fft_mlp100_llama3.1_8b_lora
#EXP_NAME=stage0.5_whisper3_freeze_clap_fft_mlp100_gemma2_9b_freeze
#EXP_NAME=debug

#EXP_NAME=stage2_only_whisper3_fft_mlp100_gemma2_9b_lora
#EXP_NAME=stage2_only_whisper3_fft_mlp100_sealion3_9b_lora
#EXP_NAME=stage2_only_whisper3_fft_mlp100_sealion3_9b_lora_less_specaug
#EXP_NAME=stage2_only_whisper3_fft_mlp100_sealion3_9b_lora_no_specaug
#EXP_NAME=stage2_only_whisper3_fft_mlp100_sealion3_9b_lora_less_specaug_short_chat_template
EXP_NAME=stage0_whisper3_fft_mlp100_sealion3_9b_freeze.yaml
EXP_NAME=debug

MASTER_NODE=$(head -n 1 $PBS_NODEFILE)
NNODES=$(cat $PBS_NODEFILE | wc -l)

echo $MASTER_NODE
echo $NNODES
pbsdsh bash $PROJ_DIR/scripts/a2ap_scripts/enroot_scripts/enroot_start.sh $ROOT_DIR $MASTER_NODE $NNODES $EXP_NAME $STAGE
#pbsdsh bash $PROJ_DIR/scripts/a2ap_scripts/enroot_scripts/enroot_start_mds_conversion.sh $ROOT_DIR $MASTER_NODE $NNODES $EXP_NAME $STAGE
