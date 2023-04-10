#! /bin/bash
set -u

[[ -z ${SPARKJOB_JOBID+X} ]] &&
	declare SPARKJOB_JOBID=$PBS_JOBID    # Change it for other job system
export SPARKJOB_JOBID
echo "SPARKJOB_JOBID is $SPARKJOB_JOBID"
echo "script dir: $SPARKJOB_SCRIPTS_DIR"
# Set the directory containing our scripts if unset.
# SPARKJOB_SCRIPTS_DIR is passed to the job via qsub.
[[ -z ${SPARKJOB_SCRIPTS_DIR+X} ]] &&
	declare SPARKJOB_SCRIPTS_DIR="$(cd $(dirname "$0")&&pwd)"
export SPARKJOB_SCRIPTS_DIR
[[ -z ${SPARKJOB_CONFIG_DIR+X} ]] &&
        declare SPARKJOB_CONFIG_DIR="$(pwd)"
export SPARKJOB_CONFIG_DIR
[[ -z ${SPARKJOB_OUTPUT_DIR+X} ]] &&
	declare SPARKJOB_OUTPUT_DIR="$(pwd)"
export SPARKJOB_OUTPUT_DIR

source "$SPARKJOB_SCRIPTS_DIR/setup.sh"
echo "sourced $SPARKJOB_SCRIPTS_DIR/setup.sh"

[[ -d $SPARK_WORKER_DIR ]] || mkdir -p "$SPARK_WORKER_DIR"
[[ -d $SPARK_CONF_DIR ]] || mkdir -p "$SPARK_CONF_DIR"
[[ -d $SPARK_LOG_DIR ]] || mkdir -p "$SPARK_LOG_DIR"

# check and convert relative path if relative path doesn't exist
to_abs_path_if_nc()
{
	if [ -e "$1" ];then
		echo "$1"
	elif [[ "$1" = /* ]];then
		echo "file $1, doesn't exit" >&2
		echo "$1"
	else
		new_path="$SPARKJOB_CONFIG_DIR/$1"
		if [ -e "$new_path" ];then
			echo "$new_path"
		else
			echo "file $1($new_path), doesn't exit" >&2
			echo "$1"
		fi
	fi
}

export -f to_abs_path_if_nc

arg_array=( ${SPARKJOB_ARG//^/ } )

# copy jars to node's tmp folder and use "local://" file to submit
declare -a arguments=()
for p in "${arg_array[@]}"
do
        if [[ "$p" =~ .*\.jar ]]; then
		fp=$(to_abs_path_if_nc "$p")
                if [[ ! -f "$fp" ]]; then
                        echo "Error: jar not exists, $fp"
                        exit 1
                fi
		while read -r line;
		do
			scp "$fp" $line:/tmp
		done < "$PBS_NODEFILE"
                arguments+=("local:///tmp/$(basename $fp)")
        else
                arguments+=("$p")
        fi
done

cp "$PBS_NODEFILE" "$SPARK_CONF_DIR/nodes"
"$SPARKJOB_SCRIPTS_DIR/run-spark.sh" "${arguments[@]}"
