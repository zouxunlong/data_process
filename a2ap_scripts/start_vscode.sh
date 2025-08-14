export project_folder=/data/projects/13003558

if [ -z "$1" ]; then
	job_id=$(qstat -u $USER | awk 'NR>5 {print $1}' | head -n 1)
  else
	job_id=$1.pbs111
fi


export PBS_JOBID=$job_id
node=$(qstat -f $job_id | grep exec_host | awk -F'=' '{print $2}' | awk -F'+' '{print $1}' | awk -F'/' '{print $1}')
echo $node
ssh -t $node bash --login -c '"
	enroot start --rw \
	-m /data/projects/13003558:/project_folder \
	multimodal_trainer bash -c \"cp /project_folder/code . && ./code tunnel --cli-data-dir ~/.vscode --log debug --accept-server-license-terms --verbose\" 
	"'