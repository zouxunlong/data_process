Host nvidia
  HostName tp1-jb.frontier.ngc.nvidia.com
  Port 2222
  user zoux
 
Host taipei1
  HostName 198.100.173.158
  user zoux
  ProxyJump nvidia
  ForwardAgent yes

  
ssh -J wang_bin@tp1-jb.frontier.ngc.nvidia.com:2222 wang_bin@198.100.173.158

srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --exclusive --exclude cnode5-004 --pty /bin/bash
srun -N 1 --gres=gpu:6 --ntasks-per-node 1 --cpus-per-task=50 --exclusive --pty /bin/bash
srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --time=00-20:00:00 --nodelist cnode5-[001,003-006] --exclusive --pty /bin/bash
srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --time=20-20:00:00 --job-name=full_long --nodelist cnode5-[001,003-006] --exclusive --pty /bin/bash
srun -N 1 --gres=gpu:0 --ntasks-per-node 1 --cpus-per-task=20 --time=20-00:00:00 --job-name=full_long --nodelist cnode5-[003] --pty /bin/bash

srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --time=20-00:00:00 --job-name=full_long --exclusive --pty /bin/bash

srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --exclude cnode5-[001,003-006] --time=10-00:00:00 --exclusive --pty /bin/bash
srun -N 1 --gres=gpu:4 --ntasks-per-node 1 --cpus-per-task=50 --pty /bin/bash
srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --time=30-00:00:00 --exclusive --nodelist cnode5-[004] --pty /bin/bash
srun -N 1 --gres=gpu:8 --ntasks-per-node 1 --cpus-per-task=200 --time=30-00:00:00 --pty /bin/bash

srun -N 1 --gres=gpu:4 --ntasks-per-node 1 --cpus-per-task=200 --time=10-00:00:00 --job-name=halfnode --nodelist cnode5-[006] --pty /bin/bash 

srun -N 1 --gres=gpu:4 --ntasks-per-node 1 --cpus-per-task=64 --time=7-00:00:00 --job-name=4gpus --nodelist cnode5-[012] --pty /bin/bash 

/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone --sftp-ssh "ssh taipei-1" sync /data/wangbin/research/9_speech_text_data/AQA/schemed/ taipei-1:/mnt/home/wang_bin/workspaces/data_generation/AQA/ --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line

/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone --sftp-ssh "ssh taipei-1" sync /home/Collaborative_Projects/SpeechEval-Related taipei-1:/mnt/home/wang_bin/workspaces/SpeechEval-Related --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Sync from 194 to LUMI
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /home/Collaborative_Projects/SpeechEval-Related lumi:/scratch/project_462000514/wangbin/workspaces/SpeechEval-Related --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line

# Sync from LUMI to Taipei-1
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /home/Collaborative_Projects/SpeechEval-Related lumi:/scratch/project_462000514/wangbin/workspaces/SpeechEval-Related --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Multimodal datasets Sync
# On Storage to LUMI
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /home/all_datasets/datasets_multimodal lumi:/scratch/project_462000514/datasets_multimodal --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/for_data_transfer/ready_datasets /home/all_datasets/pre_ready_datasets/wangbin_working_repo/ready_datasets --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M
# From LUMI to Taipei-1
/mnt/home/wang_bin/workspaces/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/datasets_multimodal /mnt/home/wang_bin/datasets_multimodal --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line

# From Taipei-1 to LUMI
# Execute from Taipei-1
/mnt/home/wang_bin/workspaces/tools/rclone-v1.66.0-linux-amd64/rclone sync /mnt/home/wang_bin/workspaces/ready_datasets lumi:/scratch/project_462000514/wangbin/workspaces/for_data_transfer/ready_datasets --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line

lumi:/scratch/project_462000514/wangbin/workspaces/temp_data_transfer/SpeechEval-Related /mnt/home/wang_bin/workspaces/SpeechEval-Related --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
srun --overlap --pty --jobid=1962 $SHELL
enroot import nvcr.io/nvidia/tensorflow:23.03-tf1-py3
srun --ntasks=1 --container-image nvcr.io#nvidia/tensorflow:23.03-tf1-py3 --container-save tf2303.sqsh  true
srun --ntasks=1 --container-image nvcr.io#nvidia/pytorch:24.05-py3 --container-save pt2405.sqsh  true

enroot list -f

enroot start -w -m /mnt/home/zhang_wenyu:/mnt/home/zhang_wenyu audiobench_v1
enroot start -w -m /mnt/home/zoux:/mnt/home/zoux multimodal_trainer

# /mnt/home/wang_bin/workspaces/containers/base_containers/pt2405.sqsh
# 1847



apt-get update
apt-get install htop
enroot export --output /mnt/home/wang_bin/workspaces/containers/customized_containers/audiobench_v1.sqsh audiobench
enroot remove vllm

enroot create --name speecheval_v1 /mnt/home/wang_bin/workspaces/containers/base_containers/pt2405.sqsh
enroot start --root -w speecheval_v1
apt-get update
apt-get install htop

enroot export --output /mnt/home/wang_bin/workspaces/containers/customized_containers/speecheval_v2.sqsh speecheval_v1

enroot start -w -m /mnt/home/wang_bin/workspaces/shm:/dev/shm multimodal_trainer
 

enroot create --name seaeval_v1 pt2405.sqsh

# ENROOT - SeaEval
enroot create --name seaeval /mnt/home/wang_bin/workspaces/containers/customized_containers/vllm.sqsh
enroot start --root -w seaeval

# ENROOT AudioBench
enroot create --name audiobench /mnt/home/wang_bin/workspaces/containers/customized_containers/audiobench_v1.sqsh
enroot start --root -w audiobench

# ENROOT Multimodal Trainer
enroot create --name multimodalcd_trainer /mnt/home/wang_bin/workspaces/containers/customized_containers/multimodal_trainer_4_43_new.sqsh
enroot start -w -m /mnt/home/zoux:/mnt/home/zoux -m /mnt/home/wang_bin/workspaces/shm:/dev/shm multimodal_trainer


# TEMP
/mnt/home/wang_bin/workspaces/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/SeaEval-Related/prepared_models/Meta-Llama-3-70B-Instruct-hf /mnt/home/wang_bin/workspaces/prepared_models/Meta-Llama-3-70B-Instruct-hf --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line


# /mnt/home/wang_bin/workspaces/data_generation/IMDA_30_60_120_SQA_v1

/mnt/home/wang_bin/workspaces/tools/rclone-v1.66.0-linux-amd64/rclone sync /mnt/home/wang_bin/workspaces/data_generation/IMDA_30_60_120_SQA_v1 lumi:/scratch/project_462000514/wangbin/workspaces/temp_data_transfer/IMDA_30_60_120_SQA_v1 --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line


/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/temp_data_transfer/IMDA_30_60_120_SQA_v1 /home/all_datasets/pre_ready_datasets/wangbin_working_repo/IMDA_30_60_120_SQA_v1 --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M



/mnt/home/wang_bin/workspaces/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/temp_data_transfer/SpeechEval-Related /mnt/home/wang_bin/workspaces/SpeechEval-Related --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line


/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/SeaEval-Related /home/Collaborative_Projects/SpeechEval-Related --transfers 256 --checkers=128 --buffer-size 256M --progress --stats-one-line
