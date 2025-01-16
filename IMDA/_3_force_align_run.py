
for i in {45,42,40,37,35,30,25,20}; do
    nohup bash /scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align.sh 4 $i >/scratch/users/astar/ares/zoux/workspaces/data_process/IMDA/force_align_4_$i.log 2>&1 &
done

