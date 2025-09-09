#!/bin/bash
#PBS -N interactive
#PBS -l select=1:ngpus=1:container_engine=enroot
#PBS -l walltime=12:00:00
#PBS -q normal
#PBS -P 13003558
#PBS -l container_image=/data/projects/13003558/zoux/containers/customized_containers/data.sqsh
#PBS -l container_name=data
#PBS -l enroot_env_file=/data/projects/13003558/zoux/workspaces/multimodal_trainer/scripts/a2ap_scripts/enroot_scripts/env.conf
#PBS -j oe
#PBS -k oed

echo "hello"
sleep 360000

# enroot start \
# 	-m /data:/data -r -w \
# 	data \
# 	bash -c "
# 	python /data/projects/13003558/zoux/workspace/data_process/normalization/normalization_zh.py True 2> /data/projects/13003558/zoux/xl_experiment.out 1> /data/projects/13003558/zoux/xl_experiment.out
#   "
