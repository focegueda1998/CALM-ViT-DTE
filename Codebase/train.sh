#!/bin/bash

$SPARK_HOME/bin/spark-submit \
       --master spark://master:7077 \
       --driver-memory 8G \
       --driver-cores 2 \
       --conf spark.log.level=ERROR \
       --conf spark.plugins="" \
       --conf spark.driver.resource.gpu.amount=0 \
       --conf spark.driver.resource.gpu.discoveryScript=${SPARK_RAPIDS_DIR}/getGpusResources.sh \
       --conf spark.executor.memory=12G \
       --conf spark.executor.instances=8 \
       --conf spark.executor.cores=4 \
       --conf spark.task.cpus=4 \
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
       --py-files file:///config/Codebase/Vi_Tools_CNN_less.py,file:///config/Codebase/reverse_ViT_hybrid.py file:///config/Codebase/distributed_trainer.py ;