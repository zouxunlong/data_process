srun -N 1 \
     -G 8 \
     --job-name=8gpus \
     --nodelist=cnode5-[007] \
     --pty /bin/bash
