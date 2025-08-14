ROOT_DIR=$1
MASTER_NODE=$2
NNODES=$3
EXP_NAME=$4
STAGE=$5

PROJ_DIR=$ROOT_DIR/workspace/multimodal_trainer
DATASET_DIR=$ROOT_DIR/datasets/datasets_multimodal_raw

HF_HOME=$ROOT_DIR/cache/hf_cache
WANDB_CONFIG_DIR=$ROOT_DIR/cache/wandb/config
WANDB_CACHE_DIR=$ROOT_DIR/cache/wandb/cache
WANDB_DIR=$ROOT_DIR/cache/wandb/log

enroot start -r -w -m $ROOT_DIR:/home -m $DATASET_DIR:/datasets multimodal_trainer \
	python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"



if [[ "$STAGE" == "2" ]];
then
	STAGE1_EXP_NAME=${EXP_NAME}
	STAGE2_EXP_NAME=${EXP_NAME/stage1/stage2}
	output_dir=/home/checkpoints/${STAGE2_EXP_NAME}
	CMD2=" dataset='stage${STAGE}_v2_a2ap' \
		model_state_dict_path=/home/checkpoints/${STAGE1_EXP_NAME}/best/model.safetensors \
		wandb.project_name='SS_stage2_runs' \
		wandb.run_name='${STAGE2_EXP_NAME}' \
		wandb.logging=true"
else
	output_dir=/home/checkpoints/${EXP_NAME}
fi

CMD="cd $PROJ_DIR;
	torchrun \
	    --nnodes=$NNODES \
            --nproc_per_node=8 \
            --rdzv_id=$MASTER_NODE \
            --rdzv_backend=c10d \
	    --rdzv_endpoint=$MASTER_NODE:29400 \
            train.py \
                dataset_basedir=/datasets\
                mds_cache_dir=/mnt/scratch/mds_cache \
                output_dir=${output_dir} \
                +experiment=${EXP_NAME} \
		"
CMD=${CMD}${CMD2}



echo $CMD

enroot start \
	-r -w \
	-m $ROOT_DIR:/home -m $DATASET_DIR:/datasets \
	-e HF_HOME=$HF_HOME \
	-e WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
	-e WANDB_CACHE_DIR=$WANDB_CACHE_DIR \
	-e WANDB_DIR=$WANDB_DIR \
	multimodal_trainer \
	bash -c "$CMD"

