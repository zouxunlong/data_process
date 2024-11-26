for i in {0..8}; do
    nohup bash /scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align.sh 6 $i >/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align_6_$i.log 2>&1 &
done


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /scratch/users/astar/ares/zoux/workspaces/data_process/NeMo/tools/nemo_forced_aligner/align.py pretrained_name="stt_en_fastconformer_hybrid_large_pc" manifest_filepath=/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5/manifest_10.jsonl output_dir=/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART5/NFA_output additional_segment_grouping_separator="|" 