###
# Created Date: Wednesday, October 25th 2023, 12:18:46 pm
# Author: Bin Wang & Yizhou Peng
# -----

#myprojects -p personal-wangb1

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

myprojects -p 13003565

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

myquota -p 13003565

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

myusage -p 13003565

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

#pbsnodes -avs pbs102 | grep resources_available.ngpus | cut -f2 -d "=" | paste -sd+ | bc
#pbsnodes -avS -s pbs102 | grep free | cut -f2 -d "=" | paste -sd+ | bc


echo "(ai queue) Total number of GPUs:"
pbs102=`pbsnodes -avS -s pbs102 | grep "free\|job-busy" | awk '{print $10}' | paste -sd+ | bc`
pbs101=`pbsnodes -avS -s pbs101 | grep gpu | grep "free\|job-busy" | awk '{print $10}' | paste -sd+ | bc`
echo $[ $pbs101 + $pbs102 ]
echo "(ai queue) Total number of GPUs assigned:"
nums_run_ai_pbs102=$(pbsnodes -avs pbs102 | grep resources_assigned.ngpus | cut -f2 -d "=")
num_run_ai_pbs102=$(echo $nums_run_ai_pbs102 | sed 's/ /\n/g' | paste -sd+ | bc)
nums_run_ai_pbs101=$(pbsnodes -avS -s pbs101 | grep gpu | grep "free\|job-busy" | awk '{ print $1}' | xargs pbsnodes -v | grep resources_assigned.ngpus | cut -f2 -d "=")
num_run_ai_pbs101=$(echo $nums_run_ai_pbs101 | sed 's/ /\n/g' | paste -sd+ | bc)
echo $[ $num_run_ai_pbs102 + $num_run_ai_pbs101 ]

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="


echo "(normal queue) Total number of GPUs:"
pbsnodes -avS -s pbs101 | grep x1000 | grep "free\|job-busy" | awk '{print $10}' | paste -sd+ | bc

echo "(normal queue) Total number of GPUs assigned:"
nums_run_normal=`pbsnodes -avS -s pbs101 | grep x1000 | grep "free\|job-busy" | awk '{ print $1}' | xargs pbsnodes -v | grep resources_assigned.ngpus | cut -f2 -d "="`
num_run_normal=$(echo $nums_run_normal | sed 's/ /\n/g' | paste -sd+ | bc)
echo $((num_run_normal))

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="

echo $nums_run_normal | awk '{for(i=1;i<=NF;i++) a[$i]++} END {for(k in a) print"[Normal Queue] Free GPU number: " 4-k", number of nodes: "a[k]}'

echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
