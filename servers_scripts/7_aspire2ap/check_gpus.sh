qstat -Qf

#!/bin/bash
for node in $(pbsnodes -l all); do
  echo "Node: $node"
  pbsnodes -a $node | grep -E "^(state|jobs|resources_available.mem|resources_assigned.mem|resources_available.ncpus|resources_assigned.ncpus)"
  echo "-------------------------"
done



pbsnodes -a -S | awk '{ 
    node = $1; 
    cmd = "pbsnodes -a " node " | grep resources_available.ngpus | awk \x27{print $3}\x27"; 
    cmd | getline gpus; 
    close(cmd); 
    if (gpus == "") { gpus = "N/A" } 
    print $0, "| Available GPUs:", gpus 
}'
