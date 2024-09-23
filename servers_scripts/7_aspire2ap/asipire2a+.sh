git clone https://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@github.com/BinWang28/WB_scripts.git

qstat -Qf

172.20.126.110



/home/wangbin/rclone-v1.67.0-linux-386/rclone sync /home/users/astar/ares/wangb1/scratch/AudioLLMs/AudioGemma nscc:/mnt/data/wangbin_new/backup/workspaces_from_taipei/AudioGemma --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M

rsync -aP /mnt/data/wangbin_new/backup/workspaces_from_taipei/AudioGemma nscc:/home/users/astar/ares/wangb1/scratch/AudioLLMs/AudioGemma

rsync -aP /home/users/astar/ares/wangb1/scratch/AudioLLMs/AudioGemma nscc

 /home/wangbin/rclone-v1.67.0-linux-386/rclone sync nscc:/mnt/data/wangbin_new/backup/workspaces_from_taipei/AudioGemma /home/users/astar/ares/wangb1/scratch/AudioLLMs/AudioGemma --progress --stats-one-line --transfers 256 --checkers=64 --buffer-size 256M
