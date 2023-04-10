# This file can be sourced multiple times

# Set the working dir if unset, requires JOBID
if [[ -z ${SPARKJOB_WORKING_DIR+X} ]];then
        if [[ -z ${SPARKJOB_JOBID+X} ]];then
                echo "Warning: SPARKJOB_JOBID missing for setup-common.sh"
                declare SPARKJOB_WORKING_DIR="$SPARKJOB_OUTPUT_DIR"
        else
                declare SPARKJOB_WORKING_DIR="$SPARKJOB_OUTPUT_DIR/$SPARKJOB_JOBID"
        fi
fi
export SPARKJOB_WORKING_DIR

export SPARKJOB_WORKING_ENVS="$SPARKJOB_WORKING_DIR/.spark-env"

NEED_HOST=1
if [[ ($# > 0) && ($1 == 0) ]];then
	NEED_HOST=0
fi
echo "need spark job host: $NEED_HOST"
if [[ (-z ${SPARKJOB_HOST+X}) && ($NEED_HOST == 1) ]];then
	declare -x SPARKJOB_HOST=arcticus
	declare -r host=$(hostname)
	if [[ $host =~ .*arcticus* ]];then
		declare -x SPARKJOB_HOST=arcticus
	elif [[ $host =~ .*yarrow* ]];then
		declare -x SPARKJOB_HOST=yarrow
	elif [[ $host =~ .*b0n0 ]];then
		declare -x SPARKJOB_HOST=sunspot
	else
		echo "unsupported host, $host. expect either arcticus or yarrow host"
		exit 1
	fi
fi
