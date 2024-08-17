tmux new -d -s rsync_more4
for i in `seq 40 59`; do
    tmux split-window
    sleep 1
    tmux send-keys "rsync -avP zoux@aspire2a.a-star.edu.sg:/scratch/users/astar/ares/zoux/data_youtube/audios_more/$i /home/user/data/data_youtube/audios_more" Enter
    sleep 5
    tmux send-keys "1z9x8l8@Ali!" Enter
    tmux select-layout tiled
done