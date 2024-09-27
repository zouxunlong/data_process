srun -N 1 \
     -G 8 \
     --job-name=full \
     --nodelist=cnode5-[005] \
     --exclusive \
     --pty /bin/bash
