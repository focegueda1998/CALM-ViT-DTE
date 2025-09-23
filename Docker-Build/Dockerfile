# Ubuntu 22.04 with CUDA 12.4.1 and cuDNN 8.2.4,
# Would have used 24.04 but there is some minor issues with the nvidia/cuda image.
# You may choose to use a different base image at your own discretion; just be aware
# this Dockerfile is built specifically for: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Copy Hadoop and Spark tarballs to /usr/local/
# You may replace these with your desired version.
# Original sources for each version in the sources.txt file.
# Make sure to download each source and place them in the same directory as the Dockerfile.
# Ensure that the Spark version is compatible with the RAPIDS plugin version if you plan on
# using a different version Spark.
COPY hadoop-3.4.1.tar.gz /usr/local/
COPY spark-4.0.1-bin-hadoop3-connect.tgz /usr/local/
COPY kafka_2.13-4.1.0.tgz /usr/local/

# Install necessary packages and set up the environment
RUN apt-get update && \
    apt-get install -y vim curl ssh openjdk-17-jdk default-jre sudo wget && \
    apt-get clean && \
    ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    mkdir -p /usr/local/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /usr/local/miniconda3/miniconda.sh && \
    bash /usr/local/miniconda3/miniconda.sh -b -u -p /usr/local/miniconda3 && \
    rm /usr/local/miniconda3/miniconda.sh && \
   # First accept ToS for main channels before any other conda operations
    /usr/local/miniconda3/bin/conda config --set auto_activate_base false && \
    /usr/local/miniconda3/bin/conda config --set channel_priority flexible && \
    # Use non-interactive acceptance method
    echo "y" | /usr/local/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    echo "y" | /usr/local/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    # Then configure and update conda
    /usr/local/miniconda3/bin/conda config --set report_errors false && \
    /usr/local/miniconda3/bin/conda update -n base conda -y && \
    # Install packages after ToS acceptance
    /usr/local/miniconda3/bin/conda install pyspark -y && \
    /usr/local/miniconda3/bin/conda install python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y && \
    mkdir -p /usr/lib/spark/scripts/gpu/ && \
    cd /usr/lib/spark/scripts/gpu/ && \
    wget https://raw.githubusercontent.com/apache/spark/master/examples/src/main/scripts/getGpusResources.sh && \
    chmod a+rwx -R /usr/lib/spark/scripts/gpu/ && \
    # Install Hadoop and Spark
    # If you want to use a different version, replace the tarballs and the paths below.
    tar -zxvf /usr/local/hadoop-3.4.1.tar.gz -C /usr/local/ && \
    tar -zxvf /usr/local/spark-4.0.1-bin-hadoop3-connect.tgz -C /usr/local/ && \
    tar -zxvf /usr/local/kafka_2.13-4.1.0.tgz -C /usr/local/ && \
    mv /usr/local/spark-4.0.1-bin-hadoop3-connect /usr/local/spark && \
    mv /usr/local/kafka_2.13-4.1.0 /usr/local/kafka && \
    rm -rf /usr/local/hadoop-3.4.1.tar.gz && \
    rm -rf /usr/local/spark-4.0.1-bin-hadoop3-connect.tgz && \
    rm -rf /usr/local/kafka_2.13-4.1.0.tgz && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/core-site.xml && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/hdfs-site.xml && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/mapred-site.xml && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/yarn-site.xml && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/capacity-scheduler.xml && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/hadoop-env.sh && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/container-executor.cfg && \
    rm -rf /usr/local/hadoop-3.4.1/etc/hadoop/workers && \
    mkdir -p /opt/sparkRapidsPlugin && \
    mkdir -p /var/hadoop/ && \
    mkdir -p /var/hadoop/yarn && \
    mkdir -p /var/hadoop/yarn/local && \
    mkdir -p /var/hadoop/yarn/logs && \
    mkdir -p /var/kafka/ && \
    chmod a+rwx -R /var/hadoop/ && \
    chmod a+rwx -R /var/hadoop/yarn && \
    chmod a+rwx -R /var/hadoop/yarn/local && \
    chmod a+rwx -R /var/hadoop/yarn/logs && \
    chmod a+rwx -R /var/kafka/ && \
    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup -y && \
    # Make note of these as they will be used in the environment setup, and overall make navigating the container easier.
    # You may change these to your desired paths.
    echo "/etc/init.d/ssh start" >> /root/.bashrc && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/" >> /root/.bashrc && \
    echo "export PATH=\$PATH:\$JAVA_HOME/bin:/root/.local/share/coursier/bin:/usr/local/miniconda3/bin" >> /root/.bashrc && \
    echo ". /etc/profile" >> /root/.bashrc && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/" >> /etc/profile && \
    echo "export HADOOP_HOME=/usr/local/hadoop-3.4.1" >> /etc/profile && \
    echo "export PATH=\$PATH:\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin" >> /etc/profile && \
    echo "export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop" >> /etc/profile && \
    echo "export SPARK_HOME=/usr/local/spark" >> /etc/profile && \
    echo "export PYSPARK_PYTHON=/usr/local/miniconda3/bin/python"  >> /etc/profile && \
    echo "export SPARK_RAPIDS_DIR=/opt/sparkRapidsPlugin"  >> /etc/profile && \
    echo "export SPARK_RAPIDS_PLUGIN_JAR=\${SPARK_RAPIDS_DIR}/rapids-4-spark_2.13-25.08.0.jar"  >> /etc/profile && \
    . /etc/profile && \
    . /root/.bashrc && \
    # May need to change the version of Scala based on your Spark Installation
    cs install scala:2.13.15 scalac:2.13.15

# Copy Hadoop configuration files
# You may replace the contents of each file with your desired configuration.
# Make sure to place these files according to your Hadoop installation.
COPY core-site.xml /usr/local/hadoop-3.4.1/etc/hadoop/
COPY hdfs-site.xml /usr/local/hadoop-3.4.1/etc/hadoop/
COPY mapred-site.xml /usr/local/hadoop-3.4.1/etc/hadoop/
COPY yarn-site.xml /usr/local/hadoop-3.4.1/etc/hadoop/
COPY hadoop-env.sh /usr/local/hadoop-3.4.1/etc/hadoop/
COPY capacity-scheduler.xml /usr/local/hadoop-3.4.1/etc/hadoop/
COPY container-executor.cfg /usr/local/hadoop-3.4.1/etc/hadoop/
COPY workers /usr/local/hadoop-3.4.1/etc/hadoop/

# Copy Spark configuration files
# You may replace the contents of each file with your desired configuration.
# Shouln't need to change much here since the Spark path will always be the same.
COPY spark-defaults.conf /usr/local/spark/conf/
COPY spark-env.sh /usr/local/spark/conf/
COPY log4j2.properties /usr/local/spark/conf/
COPY workers /usr/local/spark/conf/

# Copy RAPIDS plugin and GPU resources script
# You may replace the jar file with your desired version,
# but ensure that the jar file is compatible with your Spark version.
COPY rapids-4-spark_2.13-25.08.0.jar /opt/sparkRapidsPlugin/
COPY getGpusResources.sh /opt/sparkRapidsPlugin/

# Expose ports
EXPOSE 4040
EXPOSE 7077
EXPOSE 7078
EXPOSE 8080
EXPOSE 8081
EXPOSE 8088
EXPOSE 9864
EXPOSE 9000
EXPOSE 9870
EXPOSE 8042
EXPOSE 9868
EXPOSE 10020
EXPOSE 19888