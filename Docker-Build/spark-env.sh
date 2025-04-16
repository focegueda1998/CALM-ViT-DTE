#!/bin/bash

HADOOP_CONF_DIR=/usr/local/hadoop-3.4.0/etc/hadoop
YARN_CONF_DIR=/usr/local/hadoop-3.4.0/etc/hadoop
export SPARK_MASTER_HOST=$(hostname)
export SPARK_MASTER_PORT=7077
export SPARK_RAPIDS_DIR=/opt/sparkRapidsPlugin
export SPARK_RAPIDS_PLUGIN_JAR=${SPARK_RAPIDS_DIR}/rapids-4-spark_2.13-24.10.1.jar
SPARK_MASTER_WEBUI_PORT=8080
SPARK_WORKER_PORT=7078
SPARK_WORKER_WEBUI_PORT=8081
SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://$(hostname):9000/sparklog/ -Dspark.history.fs.cleaner.enabled=true -Dspark.history.ui.port=18080"
