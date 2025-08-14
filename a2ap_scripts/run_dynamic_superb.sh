#!/bin/bash
#PBS -N dynamicSUPERB
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

MODEL_DIR=${PROJ_DIR}/exp/stage2_only_adaptoronly_emotion2vecpluslarge_fft_mlp100_gemma2_9b_lora_nolayerdrop/checkpoint-120000
OUT_DIR=${MODEL_DIR}/dynamicsuperb

DATASET_DIR=/data/projects/13003558/dynamic_superb_hf
DYNAMIC_SUPERB_REPO_DIR=/data/projects/13003558/dynamic-superb

HF_HOME=/data/projects/13003558/hf_cache

pbs_env=`env | grep -i "PBS_" | cut -d "=" -f 1 | sed '{:q;N;s/\n/ -e /g;t q}'`

enroot start \
	-m /data:/data -m /scratch:/scratch \
  -e $pbs_env \
	-e HF_HOME=$HF_HOME \
	multimodal_trainer \
	bash -c "
	cd $PROJ_DIR
  mkdir -p $OUT_DIR
  if [ ! -f ${MODEL_DIR}/model.safetensors ]; then
    python scripts/other_scripts/convert_dcp_to_safetensors.py $MODEL_DIR
  fi
	python eval_dynamic_superb.py --out-dir $OUT_DIR --data-dir $DATASET_DIR --dynamic-superb-repo-dir $DYNAMIC_SUPERB_REPO_DIR --model-dir $MODEL_DIR --skip-done true 2> ${OUT_DIR}/stderr.txt 1> ${OUT_DIR}/stdout.txt &&
  python scripts/jeremy_scripts/collate_dynamic_superb_scores.py --out-dir $OUT_DIR --scores-json ${OUT_DIR}/scores.json"
