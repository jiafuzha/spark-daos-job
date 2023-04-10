# set components path
export APPS_DIR=/lus/gila/projects/Aurora_deployment/spark/
# export DAOS_ROOT=/home/kalfizah/daos-2tb2/install
# HOME directories
# SPARK
export SPARK_HOME=$APPS_DIR/spark
# HADOOP
export HADOOP_HOME=$APPS_DIR/hadoop
# JAVA
export JAVA_HOME=$APPS_DIR/java

[[ -z ${SPARKJOB_OUTPUT_DIR+X} ]] && declare SPARKJOB_OUTPUT_DIR="$(pwd)"
[[ -z ${SPARKJOB_CONFIG_DIR+X} ]] && declare SPARKJOB_CONFIG_DIR="$(pwd)"

export CLASSPATH=$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$JAVA_HOME/jre/lib/rt.jar

export HADOOP_LIB_DIR=${HADOOP_HOME}/lib
export HADOOP_LIBEXEC_DIR=${HADOOP_HOME}/libexec
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
export HADOOP_CLASSPATH=${HADOOP_CLASSPATH+}:$SPARKJOB_CONFIG_DIR
export HADOOP_YARN_USER=$USER
export HADOOP_USER_NAME=$USER
# PATH
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH
# TODO: python 2 or 3

# LOAD DAOS and STARTUP DAOS AGENT
SCRIPT_PATH=$(dirname "$BASH_SOURCE")

[[ -z ${SPARKJOB_SCRIPTS_DIR+X} ]] \
        && declare SPARKJOB_SCRIPTS_DIR="$(cd ${SCRIPT_PATH}&&pwd)"

source $SPARKJOB_SCRIPTS_DIR/setup-common.sh

# MORE SPARK DIR
export SPARK_WORKER_DIR="/tmp/workers-${USER}"
[[ -z ${SPARKJOB_WORKING_DIR+X} ]] && declare SPARKJOB_WORKING_DIR="$(pwd)"
export SPARK_CONF_DIR="$SPARKJOB_WORKING_DIR/conf"
export SPARK_LOG_DIR="$SPARKJOB_WORKING_DIR/logs"

source $SPARKJOB_CONFIG_DIR/env_$SPARKJOB_HOST.sh
echo "sourced $SPARKJOB_CONFIG_DIR/env_$SPARKJOB_HOST.sh"


agent_started=$(ps -ef | grep daos_agent | grep -v grep)
if [ "$SPARKJOB_DAOS" -gt 0 ] && [ -z "$agent_started" ]; then
		mkdir -p $DAOS_AGENT_DIR
		daos_agent -i -d  -o $DAOS_AGENT_CONF -l $DAOS_AGENT_LOG -s ${DAOS_AGENT_DIR} > $DAOS_AGENT_DIR/1.log 2>&1 &
		if (($?!=0)); then
			echo "failed to start daos_agent with config file, $DAOS_AGENT_CONF"
			return 1
		fi
		sleep .5
		agent_started=$(ps -ef | grep daos_agent | grep -v grep)
		if [ -z "$agent_started" ]; then
			echo "failed to start daos_agent with config file, $DAOS_AGENT_CONF"
			return 1
		fi
fi
