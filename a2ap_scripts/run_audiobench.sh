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

AUDIOBENCH_REPO_DIR=/data/projects/13003558/audiobench

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
  python ${AUDIOBENCH_REPO_DIR}/src/main_evaluate.py --dataset_name $DATASET --model_name $MODEL_TYPE --batch_size 1 --overwrite False --metrics $METRIC --number_of_samples -1 --multimodal_trainer_path $PROJ_DIR --checkpoint $MODEL_DIR --out_dir $OUT_DIR 2> ${OUT_DIR}/${DATASET}.stderr.txt 1> ${OUT_DIR}/${DATASET}.stdout.txt
  "
