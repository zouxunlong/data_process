
/home/wangbin/rclone-v1.67.0-linux-386/rclone sync lumi:/scratch/project_462000514/wangbin/workspaces_from_taipei /mnt/data/wangbin_new/backup/workspaces_from_taipei --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M


./rclone sync /home/all_datasets/datasets_multimodal lumi:/scratch/project_462000514/datasets_multimodal2 --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M


./rclone sync /mnt/data/wangbin/workspaces/AudioGemma nscc2:/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioGemma --progress --stats-one-line --transfers 1024 --checkers=64 --buffer-size 32M


/home/wangbin/rclone-v1.67.0-linux-386/rclone sync /mnt/data/wangbin/workspaces/AudioGemma nscc2:/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioGemma --progress --stats-one-line --transfers 64 --checkers=32 --buffer-size 32M
/home/wangbin/rclone-v1.67.0-linux-386/rclone sync /mnt/data/wangbin/workspaces/AudioBench_private nscc2:/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioBench_private --progress --stats-one-line --transfers 64 --checkers=32 --buffer-size 32M
/home/wangbin/rclone-v1.67.0-linux-386/rclone sync /mnt/data/wangbin/workspaces/containers nscc2:/home/users/astar/ares/wangb1/scratch/workspaces_wb/containers --progress --stats-one-line --transfers 64 --checkers=32 --buffer-size 32M
/home/wangbin/rclone-v1.67.0-linux-386/rclone sync /mnt/data/wangbin/workspaces/SeaEval nscc2:/home/users/astar/ares/wangb1/scratch/workspaces_wb/SeaEval --progress --stats-one-line --transfers 64 --checkers=32 --buffer-size 32M
