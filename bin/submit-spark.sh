#! /bin/bash
set -u

# Set the directory containing our scripts if unset.
[[ -z ${SPARKJOB_SCRIPTS_DIR+X} ]] && \
	declare SPARKJOB_SCRIPTS_DIR="$(cd $(dirname "$0")&&pwd)"

declare -r configdir="$(pwd)"
declare -r version='Spark DAOS Job  v1.0.3 for Sunspot'
declare -r usage="$version"'

Usage:
	submit-spark.sh [options] JOBFILE [arguments ...]

JOBFILE can be:
	script.py			pyspark scripts
	run-example examplename		run Spark examples, like "SparkPi"
	--class classname example.jar	run Spark job "classname" wrapped in example.jar
	shell-script.sh			when option "-s" is specified

Required options:  
	-l walltime=<v>		Max run time, e.g., -l walltime=30:00 for running 30 minutes at most
	-l select=<v>		Job node count, e.g., -l select=2 for requesting 2 nodes
	-q QUEUE		Queue name

	Optional options:
	-A PROJECT		Allocation name
	-o OUTPUTDIR		Directory for job output files (default: current dir)
	-s			Enable shell script mode
	-y			Spark master uses a separate node
	-I			Start an interactive ssh session
	-b			Prefer Spark built-in mllib to default OAP mllib 
	-n			Run without DAOS loaded
	-h			Print this help messages
Example:
	submit-spark.sh -A Aurora_deployment -l walltime=60 -l select=2 -q workq dense_kmeans_example.py daos://pool0/cont1/kmeans/input/libsvm 10
	submit-spark.sh -A Aurora_deployment -l walltime=30 -l select=1 -q workq -o output-dir run-example SparkPi
	submit-spark.sh -A Aurora_deployment -I -l walltime=30 -l select=2 -q workq --class com.intel.jlse.ml.KMeansExample example/jlse-ml-1.0-SNAPSHOT.jar daos://pool0/cont1/kmeans/input/csv csv 10
	submit-spark.sh -A Aurora_deployment -l walltime=30 -l select=1 -q workq -s example/test_script.sh job.log
'

declare -A resources

add_resource()
{
        old_ifs=$IFS
        IFS='='
        a=($1)
        IFS=$old_ifs
        al=${#a[@]}
        if ((al!=2));then
                echo "bad resource, $1"
        else
                resources[${a[0]}]=${a[1]}
        fi
}


while getopts :hysIbnA:l:w:q:o: OPT; do
	case $OPT in
	A)	declare -r	allocation="$OPTARG";;
	l)	add_resource "$OPTARG";;
	s)	declare -ir	scriptMode=1;;
	l)	declare -r	add_resource "$OPTARG";;
	q)	declare -r	queue="$OPTARG";;
	o)	declare -r	outputdir="$OPTARG";;
	y)	declare -ir	separate_master=1;;
	n)	declare -ir	no_daos=1;;
	w)	declare -ir	waittime=$((OPTARG*60));;
	b)	declare -ir	oapml=0;;
	I)	declare -ir	interactive=1;;
	h)	echo "$usage"; exit 0;;
	?)	
		if [[ $OPTARG == - ]];then
                        # Assume we see a double dash `--option` that should be passed to `spark-submit`.
                        break
                else
			echo "submit-spark.sh: illegal option: -$OPTARG"
			echo "$usage"
			exit 1
		fi
		;;
	esac
done

[[ -z ${waittime+X} ]] && declare -ir waittime=$((30*60))
[[ -z ${scriptMode+X} ]] && declare -ir scriptMode=0
[[ -z ${outputdir+X} ]] && declare -r outputdir=.
[[ -z ${separate_master+X} ]] && declare -ir separate_master=0
[[ -z ${oapml+X} ]] && declare -ir oapml=1 
[[ -z ${no_daos+X} ]] && declare -ir no_daos=0

key_time="walltime"
key_node="select"
[[ -z ${resources[$key_time]+X} ]] || declare -r time=${resources[$key_time]}
[[ -z ${resources[$key_node]+X} ]] || declare -r nodes=${resources[$key_node]}

if [[ -z ${time+X} || -z ${nodes+X} || -z ${queue+X} ]];then
	echo "$usage"
	exit 1
fi

shift $((OPTIND-1))

declare -a scripts=()

if (($#>0));then
	[[ -z ${interactive+X} ]] && declare -ir interactive=0
	scripts=( "$@" )
	echo "# Submitting job: ${scripts[@]}"
else
	[[ -z ${interactive+X} ]] && declare -ir interactive=1
	echo "Submitting an interactive job and wait for at most $waittime sec."
fi

if [[ ! -d $outputdir ]];then
	if ! mkdir "$outputdir";then
		echo "Cannot create directory: $outputdir"
		exit 1
	fi
fi

declare SPARKJOB_OUTPUT_DIR="$(cd $outputdir&&pwd)"
declare SPARKJOB_CONFIG_DIR=$configdir
declare SPARKJOB_INTERACTIVE=$interactive
declare SPARKJOB_SCRIPTMODE=$scriptMode
declare SPARKJOB_SEPARATE_MASTER=$separate_master
declare SPARKJOB_OAPML=$oapml
if [ "$no_daos" -eq 0 ];then
	declare -i SPARKJOB_DAOS=1
else
	declare -i SPARKJOB_DAOS=0
fi

declare SPARKJOB_JOBID=
declare SPARKJOB_ARG="${scripts[@]}"
mysubmit() {
	# Options to pass to qsub
	#--attrs 'enable_ssh=1'
	env_vars="SPARKJOB_SCRIPTS_DIR=$SPARKJOB_SCRIPTS_DIR","SPARKJOB_CONFIG_DIR=$SPARKJOB_CONFIG_DIR","SPARKJOB_INTERACTIVE=$SPARKJOB_INTERACTIVE","SPARKJOB_SCRIPTMODE=$SPARKJOB_SCRIPTMODE","SPARKJOB_OUTPUT_DIR=$SPARKJOB_OUTPUT_DIR","SPARKJOB_SEPARATE_MASTER=$SPARKJOB_SEPARATE_MASTER","SPARKJOB_OAPML=$SPARKJOB_OAPML","SPARKJOB_DAOS=$SPARKJOB_DAOS","SPARKJOB_ARG=$SPARKJOB_ARG"
	env_vars=${env_vars// /^}

	local -a opt=(
		-l "select=$nodes" -l "walltime=$time" -q $queue
		-v $env_vars
		-o "$SPARKJOB_OUTPUT_DIR"
		-e "$SPARKJOB_OUTPUT_DIR"
	)

	[[ ! -z ${allocation+X} ]] && opt=(-A $allocation "${opt[@]}")
	opt+=("$SPARKJOB_SCRIPTS_DIR/start-spark.sh")

	SPARKJOB_JOBID=$(qsub "${opt[@]}")
	echo "debug $SPARKJOB_JOBID"
	echo "debug ${opt[@]}"
	export SPARKJOB_JOBID
	if [[ ! -z $SPARKJOB_JOBID ]];then
		echo "# Submitted"
		echo "SPARKJOB_JOBID=$SPARKJOB_JOBID"
	else
		echo "# Submitting failed."
		exit 1
	fi
}

if ((interactive>0)); then
	cleanup(){ [[ ! -z $SPARKJOB_JOBID ]] && qdel $SPARKJOB_JOBID; }
	trap cleanup 0
	mysubmit
	declare -i mywait=1 count=0
	source "$SPARKJOB_SCRIPTS_DIR/setup-common.sh" 0
	JOB_PATH=$SPARKJOB_OUTPUT_DIR/$SPARKJOB_JOBID
	ENV_PATH=$JOB_PATH/.spark-env
	echo "Waiting for Spark to launch by checking $JOB_PATH"
	for ((count=0;count<waittime;count+=mywait));do
		[[ ( ! -d $JOB_PATH ) || ( ! -f $ENV_PATH ) ]] || break
		sleep 1 
	done
	[[ -s $SPARKJOB_WORKING_ENVS ]] || SPARKJOB_WORKING_ENVS=$ENV_PATH
	if [[ -s $SPARKJOB_WORKING_ENVS ]];then
		source "$SPARKJOB_WORKING_ENVS"
		echo "# Spark is now running (SPARKJOB_JOBID=$SPARKJOB_JOBID) on:"
		column "$SPARK_CONF_DIR/slaves" | sed 's/^/# /'
		declare -p SPARK_MASTER_URI
		declare -ar sshmaster=(ssh -o ControlMaster=no -t $MASTER_HOST)
		declare -r runbash="exec bash --rcfile <(
			echo SPARKJOB_JOBID=\'$SPARKJOB_JOBID\';
			echo SPARKJOB_SCRIPTS_DIR=\'$SPARKJOB_SCRIPTS_DIR\';
			echo SPARKJOB_INTERACTIVE=\'$SPARKJOB_INTERACTIVE\';
			echo SPARKJOB_SCRIPTMODE=\'$SPARKJOB_SCRIPTMODE\';
			echo SPARKJOB_OUTPUT_DIR=\'$SPARKJOB_OUTPUT_DIR\';
			echo SPARKJOB_SEPARATE_MASTER=\'$SPARKJOB_SEPARATE_MASTER\';
			echo SPARKJOB_OAPML=\'$SPARKJOB_OAPML\';
			echo SPARKJOB_DAOS=\'$SPARKJOB_DAOS\';
			echo SPARKJOB_CONFIG_DIR=\'$SPARKJOB_CONFIG_DIR\';
			echo source ~/.bashrc;
			echo source \'$SPARKJOB_SCRIPTS_DIR/setup.sh\';
			echo cd \'$SPARKJOB_OUTPUT_DIR\';
		        echo mkdir -p $SPARKJOB_OUTPUT_DIR/$SPARKJOB_JOBID/loggedin	) -i"
		echo "# Spawning bash on host: $MASTER_HOST"

		"${sshmaster[@]}" "$runbash"
		rm -rf $SPARKJOB_OUTPUT_DIR/$SPARKJOB_JOBID/loggedin	
	else
		echo "Spark failed to launch within $((waittime/60)) minutes."
	fi
else
	mysubmit
fi
