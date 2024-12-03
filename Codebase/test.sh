#!/bin/bash
$SPARK_HOME/bin/spark-submit \
       --master yarn \
       --conf spark.rapids.sql.concurrentGpuTasks=1 \
       --driver-memory 2G \
       --conf spark.driver.resource.gpu.amount=1 \
       --conf spark.driver.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --conf spark.executor.memory=20G \
       --conf spark.executor.instances=3 \
       --conf spark.executor.cores=10 \
       --conf spark.task.cpus=1 \
       --conf spark.rpc.io.serverThreads=10 \
       --conf spark.task.resource.gpu.amount=1 \
       --conf spark.rapids.memory.pinnedPool.size=2G \
       --conf spark.sql.files.maxPartitionBytes=512m \
       --conf spark.plugins=com.nvidia.spark.SQLPlugin \
       --conf spark.executor.resource.gpu.amount=1 \
       --conf spark.executor.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --files ${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --jars ${SPARK_RAPIDS_PLUGIN_JAR} \
       --py-files py_files.zip \
       MNIST_SPARK_DIST.py