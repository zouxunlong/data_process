
for part in {3,}; do
    echo "Processing part $part"
    for split in {0..7}; do
        nohup bash /scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align.sh $part $split > /scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align_part${part}_split${split}.log 2>&1 &
    done
done

