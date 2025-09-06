#!/bin/bash
#PBS -N interactive
#PBS -l select=1:ngpus=1:mem=1887gb:container_engine=enroot
#PBS -l walltime=12:00:00
#PBS -q normal
#PBS -P 13003558
#PBS -l container_image=/data/projects/13003558/zoux/containers/customized_containers/audio_bench.sqsh
#PBS -l container_name=data
#PBS -l enroot_env_file=/data/projects/13003558/zoux/workspaces/multimodal_trainer/scripts/a2ap_scripts/enroot_scripts/env.conf
#PBS -o /data/projects/13003558/zoux/workspaces/logs/stdout.txt
#PBS -e /data/projects/13003558/zoux/workspaces/logs/stderr.txt

ROOT_DIR=/data/projects/13003558/zoux
PROJ_DIR=$ROOT_DIR/workspaces/multimodal_trainer
DATASET_DIR=$ROOT_DIR/datasets

HF_HOME=/data/projects/13003558/zoux/huggingface
WANDB_CONFIG_DIR=/data/projects/13003558/zoux/wandb/config
WANDB_CACHE_DIR=/data/projects/13003558/zoux/wandb/cache
WANDB_DIR=/data/projects/13003558/zoux/wandb/logs

enroot start \
        -m $ROOT_DIR:/home -m $DATASET_DIR:/datasets \
        -e HF_HOME=$HF_HOME \
        -e WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
        -e WANDB_CACHE_DIR=$WANDB_CACHE_DIR \
        -e WANDB_DIR=$WANDB_DIR \
        -r -w \
        multimodal_trainer \
