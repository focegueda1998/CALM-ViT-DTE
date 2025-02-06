#!/bin/bash

$SPARK_HOME/bin/spark-submit \
       --master yarn \
       --driver-memory 2G \
       --conf spark.log.level=ERROR \
       --conf spark.driver.resource.gpu.amount=0 \
       --conf spark.driver.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --conf spark.executor.memory=20G \
       --conf spark.executor.instances=4 \
       --conf spark.rapids.memory.gpu.allocFraction=0.4 \
       --conf spark.rapids.memory.gpu.maxAllocFraction=1 \
       --conf spark.executor.cores=8 \
       --conf spark.task.cpus=8 \
       --conf spark.files.overwrite=true \
       --conf spark.rpc.io.serverThreads=70 \
       --conf spark.rpc.io.clientThreads=10 \
       --conf spark.task.resource.gpu.amount=1 \
       --conf spark.rapids.memory.pinnedPool.size=2G \
       --conf spark.sql.files.maxPartitionBytes=512m \
       --conf spark.plugins=com.nvidia.spark.SQLPlugin \
       --conf spark.executor.resource.gpu.amount=1 \
       --conf spark.driver.maxResultSize=5G \
       --conf spark.sql.hive.filesourcePartitionFileCacheSize=4096000000 \
       --conf spark.executor.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --conf spark.executor.extraClassPath=${SPARK_RAPIDS_PLUGIN_JAR} \
       --conf spark.driver.extraClassPath=${SPARK_RAPIDS_PLUGIN_JAR} \
       --files ${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --jars ${SPARK_RAPIDS_PLUGIN_JAR} \
       batch-to-image.py;