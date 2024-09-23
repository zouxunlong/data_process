


# myquota

# myusage

# qsub 1-submit-serial.pbs

# qsub -I -l select=1:ncpus=1:mem=12g -l walltime=01:00:00 -P 13003565 -q normal

pbsnodes -a

pbsnodes -l free

qstat -q ai

qstat -H

qstat -x

qdel <jobid>

cd /home/users/astar/ares/wangb1

cd /data/projects/13003565/wangb1

scp -r fastchat-t5-3b-v1.0 wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/prepared_models
scp -r chatglm3-6b wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/prepared_models

scp -r baichuan-2-7b wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/models
scp -r baichuan-2-7b-chat wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/models
scp -r baichuan-2-13b wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/models

scp -r ali-t5-large-061123 wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/prepared_models

scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/Instructor_single/outputs/vicuna_new_instructions ./

scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log ./

scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log ./

scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log_predictions ./

scp -r eval_data wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/

scp -r seallama-040923 wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/models/

scp -r eval_data_v1.1_unshuffled wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/

scp -r eval_data_v1.1_unshuffled wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/

scp -r all_data wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v1.0
scp -r config wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v1.0
scp -r src wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v1.0

scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003558/pretrain_output_instruct_results/checkpoint-3350-fp32 ./
/home/shared_projects/sea_lm_eval/seaeval_v0.0/eval_data_v1.1_unshuffled
/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log


scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log_predictions ./

scp -r vicuna-33b-v1.3 wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/Instructor_single/local_models



scp -r data wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/Instructor_all

rsync -avh source/ dest/ --delete


find . -name "*.log" -type f -print0 | xargs -0 rm

qsub -I -l select=1:ncpus=4:mem=12g:ngpus=4 -l walltime=24:00:00 -P 13003565 -q normal

qsub -I -l select=1:ncpus=30:mem=30g:ngpus=2 -l walltime=12:00:00 -P 13003565 -q ai

qsub -I -l select=1:ncpus=4:mem=30g:ngpus=2 -l walltime=12:00:00 -P 13003565 -q ai

qsub -I -l select=1:ncpus=4:mem=30g:ngpus=2 -l walltime=120:00:00 -P 13003565 -q ai

qsub -I -l select=1:ncpus=4:mem=50g:ngpus=4 -l walltime=24:00:00 -P 13003565 -q ai

qsub -I -l select=1:ncpus=4:mem=30g:ngpus=4 -l walltime=24:00:00 -P 13003565 -q normal

qsub -I -l select=1:ncpus=4:mem=110g:ngpus=1 -l walltime=24:00:00 -P 13003565 -q normal

qsub -I -l select=1:ncpus=4:mem=110g:ngpus=1 -l walltime=2:00:00 -P 13003565 -q normal

qsub -I -l select=1:ncpus=4:mem=50g:ngpus=1 -l walltime=2:00:00 -P 13003565 -q normal

qsub -I -l select=1:ncpus=4:mem=50g:ngpus=1 -l walltime=2:00:00 -P personal-wangb1 -q normal

qsub -I -l select=1:ncpus=4:mem=110g:ngpus=4:host=thenodename -l walltime=4:00:00 -P 13003565 -q ic005

qsub -I -l select=1:ncpus=4:mem=110g:ngpus=4 -l walltime=700:00:00 -P 13003821 -q ic005


qstat -answ @pbs102


qstat -x @pbs102

# Delete all Jobs of wangb1
qselect -u wangb1 | xargs qdel

qselect -q @pbs102 -u wangb1

qselect -H -u wangb1 

myusage -p 13003565

myquota -p 13003565
myquota -p 13003558

myprojects -p 13003565


scp -r xxx wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/temp

pbsnodes -avS -s pbs102

pbsnodes -avs pbs102

qmgr -c 'p q g1'
qmgr -c 'p q ic005'

qmgr -c 'p q aiq1' pbs102


find . -name "*.log" -type f -print0 | xargs -0 rm


13003558


scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log ./


scp -r wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/seaeval_v0.0/log_predictions ./


# Sync files from local to remote
rsync -azP prepared_models wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/prepared_models 


# Sync files from remote to local
rsync -azP wangb1@aspire2a.a-star.edu.sg:/data/projects/13003565/wangb1/research/sea_lm_eval/prepared_models prepared_models


rsync -azP research /data/wangbin/research


--delete




export PBS_JOBID=7783981.pbs101
PBS_JOBID=7904117.pbs101 ssh asp2a-gpu014




pbsnodes -avS -s pbs101 | grep ic005


singularity shell -B /data/projects/13003821/wangb1:/data/projects/13003821/wangb1 pytorch_23.06.sif

singularity shell -B /data/projects/13003821/wangb1:/data/projects/13003821/wangb1 pytorch_23.06.sif

singularity build pytorch_23.06_2.sif pytorch_23.06_2

singularity shell --writable -nv audiobench_v1

singularity build --sandbox audiobench_v2 pytorch-nvidia-22.04-py3.sif


singularity shell -B /data/projects/13003821/wangb1:/data/projects/13003821/wangb1 audiobench_v2

singularity shell -B /data/projects/13003821/wangb1:/data/projects/13003821/wangb1 --writable audiobench_v2


-B /data/projects/13003821/wangb1/AudioBench-Related:/data/projects/13003821/wangb1/AudioBench-Related

singularity --debug shell --writable-tmpfs --bind /data/projects/13003821/wangb1:/mnt --nv audiobench_v2


singularity shell --writable --bind /data/projects/13003821/wangb1:/mnt --nv audiobench_v2


singularity shell --bind /data/projects/13003821/wangb1:/mnt --nv audiobench_v2



/home/users/astar/ares/wangb1/rclone-v1.67.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces_from_taipei/AudioGemma /data/projects/13003565/wangb1/workspaces/AudioGemma --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M

singularity shell -B /home/users/astar/ares/wangb1/scratch/AudioLLMs:/mnt AudioGemma_singularity

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat/lib.real/

singularity shell -B /home/users/astar/ares/wangb1/scratch/AudioLLMs:/mnt --writable AudioGemma_singularity

singularity shell -B /home/users/astar/ares/wangb1/scratch/AudioLLMs:/mnt --nv AudioGemma_singularity


pip install flash_attn --no-build-isolation
