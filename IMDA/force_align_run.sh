
part=$1

# for i in {40..79}; do
for i in {67,}; do
    # echo $i
    nohup bash /scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align.sh $part $i > /scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align_${part}_$i.log 2>&1 &
done

