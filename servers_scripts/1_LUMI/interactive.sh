
# Ask for standard-g
salloc --job-name=trial --partition=standard-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --time=2-00:00:00 --mem=110G --cpus-per-task=16 --account=project_462000514
srun --partition=small-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=8 --pty bash

# Ask for small-g
salloc --job-name=seaeval --partition=small-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --time=2-00:00:00 --mem=300G --cpus-per-task=16 --account=project_462000514
srun --partition=small-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --pty bash

# Only for debug
salloc --job-name=debug --partition=dev-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --time=0-03:00:00 --mem=480G --cpus-per-task=56 --account=project_462000514
srun --partition=dev-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --pty bash

singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/SeaEval-Related/SeaEval_dev/singularity_seaeval.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/attn2.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 -B /scratch/project_462000514/wangbin/workspaces/SeaEval_dev/LUMI_helper/conda_hx_patched:/conda_hx_patched /scratch/project_462000514/wangbin/attn2.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_craft_v1.sif

singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_seaeval_v3.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_seaeval_v4.sif

singularity shell -B ~/.ssh:/users/wangbin/.ssh -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_gitlfs.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/noisytext.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/seaeval2.sif
singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_gitlfs.sif

singularity shell -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_seaeval_v4.sif


# /scratch/project_462000514/wangbin/workspaces/container

watch "sprio -p small-g -S -y"
sprio -p small-g -S -y -o "%i %PAR %u %j"
sprio -p small-g -S -y -o "%i %y %u %j %f %p %a"
sprio -p small-g -S -y -o "%i %u %y"

watch -n 5 'sinfo -o "%P %T %D" | grep idle'
watch "sinfo -s"

cd /scratch/project_462000514/wangbin/workspaces


srun --overlap --pty --jobid=7884668 $SHELL
watch 'rocm-smi'
watch 'rocm-smi --showmemuse --showmeminfo vram -b'


source activate LUMI_helper/conda_hx_patched/
source activate /conda_hx_patched


watch squeue -u wangbin


rclone sync --interactive prepared_models lumi:/scratch/project_462000514/wangbin/workspaces/prepared_models

rsync -aP prepared_models lumi:/scratch/project_462000514/wangbin/workspaces/

rsync -aP NoisyText lumi:/scratch/project_462000514/wangbin/workspaces/

python -m git_lfs -v git@hf.co:meta-llama/Llama-2-7b-chat-hf ./

/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk1 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk4 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b/chunk4  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk5 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b/chunk5  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk6 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b/chunk6  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk7 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b/chunk7  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk8 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b/chunk8  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /data/llm_team/wangbin_initial_trial/slimpajama-627b_reformat/train/chunk9 lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT/DATASET/Slimpajama-627b/chunk9  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync /home/wangbin/workspaces/SlimPajama-627B.en.hf/ lumi:/scratch/project_462000514/wangbin/workspaces/CRAFT-Related/data  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M


/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/SeaEval-Related/prepared_models/ /home/Collaborative_Projects/SeaEval-Related/prepared_models/  --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M

/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/SeaEval-Related /home/Collaborative_Projects/SeaEval-Related  --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M

/home/wangbin/tools/rclone-v1.66.0-linux-amd64/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces/10_CRAFT /data/wangbin/research/10_CRAFT  --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M

/scratch/project_462000514/wangbin/workspaces/SeaEval-Related

./rclone sync /home/all_datasets/datasets_multimodal lumi:/scratch/project_462000514/datasets_multimodal2 --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M