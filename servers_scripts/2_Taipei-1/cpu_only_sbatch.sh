#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=cpu-only
#SBATCH --cpus-per-task=100
#SBATCH -o %j-%N-GR.NR.MIX.out
#SBATCH -e %j-%N-GR.NR.MIX.out

container_image="/mnt/home/zoux/containers/env_data.sqsh"


echo -o "Container using ${container_image}"
echo -o "Running on hosts: $(hostname)"

# command
srun --container-image $container_image \
    --container-writable \
    bash -c "python /mnt/home/zoux/datasets/xunlong_working_repo/IMDA_scripts/build.imda.GR.NR.MIX.py"

