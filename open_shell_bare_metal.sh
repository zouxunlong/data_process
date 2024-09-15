
task_id=$1

srun --overlap --jobid $task_id --pty $SHELL