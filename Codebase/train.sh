#!/bin/bash

$SPARK_HOME/bin/spark-submit \
       --master spark://master:7077 \
       --driver-memory 8G \
       --conf spark.log.level=ERROR \
       --conf spark.plugins="" \
       --conf spark.driver.resource.gpu.amount=0 \
       --conf spark.driver.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --conf spark.driver.cores=6 \
       --conf spark.executor.memory=11G \
       --conf spark.executor.instances=8 \
       --conf spark.executor.cores=3 \
       --conf spark.task.cpus=3 \
       --conf spark.files.overwrite=true \
       --conf spark.task.resource.gpu.amount=1 \
       --conf spark.eventLog.enabled=false \
       --conf spark.sql.files.maxPartitionBytes=512m \
       --conf spark.executor.resource.gpu.amount=1 \
       --conf spark.driver.maxResultSize=10G \
       --conf spark.sql.hive.filesourcePartitionFileCacheSize=4096000000 \
       --conf spark.executor.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --conf spark.scheduler.barrier.maxConcurrentTasksCheck.maxFailures=1 \
       --files ${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --py-files file:///config/Codebase/reverse_ViT_hybrid.py \
       /config/Codebase/distributed_trainer.py;