# This file must be compatible with bash(1) on the system.
# It should be alright to source this file multiple times.

# Because you can change SPARK_CONF_DIR in this file, the main
# scripts only create the directory after sourcing this file.  Since
# we need the directory to create the spark-defaults.conf file, we
# create the directory here.
[[ -d $SPARK_CONF_DIR ]] || mkdir -p "$SPARK_CONF_DIR"

SPARKJOB_FILES=

if ((SPARKJOB_OAPML>0)); then
	SPARKJOB_FILES=${SPARKJOB_ADD_JARS//:/,}	
	# GPU options. variables defined in env_<node>.sh
    	OPTIONS="--conf spark.oap.mllib.device=GPU"
        export GPU_OPTIONS=$OPTIONS
        echo "GPU Options: $GPU_OPTIONS"
fi

# create event log dir
event_log_dir=$SPARKJOB_CONFIG_DIR/event_logs
[[ -d $event_log_dir ]] || mkdir -p "$event_log_dir"

# The created spark-defaults.conf file will only affect spark
# submitted under the current directory where this file resides.
# The parameters here may require tuning depending on the machine and workload.
[[ -s $SPARK_CONF_DIR/spark-defaults.conf ]] ||
	cat > "$SPARK_CONF_DIR/spark-defaults.conf" <<EOF
spark.executor.cores                 34
spark.driver.memory                  20g
spark.executor.memory                150g
spark.executor.instances             12

spark.worker.resourcesFile	     $GPU_RESOURCE_FILE
spark.worker.resource.gpu.amount     $GPU_WORKER_AMOUNT
spark.executor.resource.gpu.amount   1
spark.task.resource.gpu.amount	     0.01

spark.driver.extraJavaOptions        -XX:+UseG1GC
spark.executor.extraJavaOptions      -XX:+UseG1GC

spark.executor.extraClassPath	     $SPARKJOB_CONFIG_DIR:$SPARKJOB_ADD_JARS
spark.driver.extraClassPath          $SPARKJOB_CONFIG_DIR:$SPARKJOB_ADD_JARS


spark.eventLog.enabled               true
spark.eventLog.dir                   file://$event_log_dir

EOF

# TODO: python 2 or 3

# On cooley, interactive spark jobs setup ipython notebook by
# defaults.  You can change it here, along with setting up your
# other python environment.
unset PYSPARK_DRIVER_PYTHON
unset PYSPARK_DRIVER_PYTHON_OPTS
