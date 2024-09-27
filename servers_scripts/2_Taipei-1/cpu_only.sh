srun -N 1 \
     --job-name=cpu-only \
     --cpus-per-task=100 \
     --pty /bin/bash
