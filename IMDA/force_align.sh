

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

part=$1
split=$2
gpu=`expr $split % 8`

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /scratch/users/astar/ares/zoux/workspaces/data_process/NeMo/tools/nemo_forced_aligner/align.py pretrained_name="stt_en_fastconformer_hybrid_large_pc" manifest_filepath=/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART$part/manifest_$split.jsonl output_dir=/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART$part/NFA_output additional_segment_grouping_separator="|" transcribe_device="cuda:$gpu" viterbi_device="cuda:$gpu"
