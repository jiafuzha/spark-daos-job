
# set Spark Worker resources
export SPARK_WORKER_CORES=204
export SPARK_WORKER_MEMORY=900G

# set GPU options
export GPU_RESOURCE_FILE=$SPARKJOB_CONFIG_DIR/gpuResourceFile_sunspot.json
export GPU_WORKER_AMOUNT=6

