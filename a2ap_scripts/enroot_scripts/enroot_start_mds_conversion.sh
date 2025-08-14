ROOT_DIR=$1
MASTER_NODE=$2
NNODES=$3
EXP_NAME=$4

PROJ_DIR=$ROOT_DIR/workspace/multimodal_trainer
DATASET_DIR=$ROOT_DIR/datasets

enroot start \
	-r -w \
	-m $ROOT_DIR:/home -m $DATASET_DIR:/datasets \
	multimodal_trainer \
	bash -c "
	cd $PROJ_DIR/scripts/mds_helper_scripts
	torchrun \
	    --nnodes=$NNODES \
            --nproc_per_node=48 \
            --rdzv_id=$MASTER_NODE \
            --rdzv_backend=c10d \
	    --rdzv_endpoint=$MASTER_NODE:29400 \
            convert_hf_to_mds.py"
