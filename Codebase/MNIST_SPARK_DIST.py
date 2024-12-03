import CNN_TOOLS as CT
import subprocess
import pyspark
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pyspark import SparkContext, SparkConf, BarrierTaskContext
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor

# conf = SparkConf() \
#     .setAppName("Pyspark Dist.") \
#     .setMaster("yarn") \
#     .set("spark.rapids.sql.concurrentGpuTasks", "2") \
#     .set("spark.driver.memory", "2G") \
#     .set("spark.driver.resource.gpu.amount", "1") \
#     .set("spark.driver.resource.gpu.discoveryScript", "/opt/sparkRapidsPlugin/getGpusResources.sh") \
#     .set("spark.executor.memory", "4G") \
#     .set("spark.executor.cores", "4") \
#     .set("spark.executor.instances", "2") \
#     .set("spark.task.cpus", "1") \
#     .set("spark.task.resource.gpu.amount", "1" ) \
#     .set("spark.rapids.memory.pinnedPool.size", "2G") \
#     .set("spark.sql.files.maxPartitionBytes", "512m") \
#     .set("spark.plugins", "com.nvidia.spark.SQLPlugin") \
#     .set("spark.executor.resource.gpu.amount", "1") \
#     .set("spark.executor.resource.gpu.discoveryScript", "/opt/sparkRapidsPlugin/getGpusResources.sh") \
#     .set("spark.files", "/opt/sparkRapidsPlugin/getGpusResources.sh") \
#     .set("spark.jars", "/opt/sparkRapidsPlugin/rapids-4-spark_2.13-24.10.1.jar") \

# sc = SparkContext.getOrCreate(conf=conf)
# spark = SparkSession.builder \
#     .config(conf=conf) \
#     .getOrCreate()\
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
print("SparkContext initialized.")
print(f"Application ID: {sc.applicationId}")

# Training Configuration
EPOCH_COUNT = 2         # Number of training epochs
WORKER_COUNT = 4        # Number of processes (e.g., CPU cores) for training
BATCH_SIZE = 64         # Number of samples per batch

# 本地文件路径和 HDFS 路径
local_checkpoint_file_path = "model_checkpoint.pth"
hdfs_checkpoint_dir = "/user/MNIST_Checkpoint"

# # HDFS 路径
# train_images_path = "hdfs://localhost:9000/user/MNIST/train-images-idx3-ubyte"
# train_labels_path = "hdfs://localhost:9000/user/MNIST/train-labels-idx1-ubyte"
# t10k_images_path = "hdfs://localhost:9000/user/MNIST/t10k-images-idx3-ubyte"
# t10k_labels_path = "hdfs://localhost:9000/user/MNIST/t10k-labels-idx1-ubyte"

# Base HDFS path for MNIST dataset
hdfs_base_path = "hdfs://master:9000/user/MNIST"

# MNIST dataset paths
mnist_paths = {
    "train_images": f"{hdfs_base_path}/train-images-idx3-ubyte",
    "train_labels": f"{hdfs_base_path}/train-labels-idx1-ubyte",
    "test_images": f"{hdfs_base_path}/t10k-images-idx3-ubyte",
    "test_labels": f"{hdfs_base_path}/t10k-labels-idx1-ubyte"
}

def print_versions():
    import sys
    # Define a border for the box
    border = "=" * 50

    # Print the box with version information
    print(border)
    print(f"Python version: {sys.version}")
    print(f"PySpark version: {pyspark.__version__}")
    # Print Hadoop version
    try:
        hadoop_version_output = subprocess.check_output(["hadoop", "version"]).decode("utf-8")
        # Extract the first line (typically contains version info)
        hadoop_version_line = hadoop_version_output.splitlines()[0]
        print(f"Hadoop version: {hadoop_version_line}")
    except FileNotFoundError:
        print("Hadoop is not installed or not in the PATH.")

    # Print closing border
    print(border)

def generate_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    from datetime import datetime
    """Generate the current timestamp, with a default format '%Y%m%d_%H%M%S'."""
    return datetime.now().strftime(format)

def generate_checkpoint_filename(prefix: str = "model_checkpoint", timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """Generate a checkpoint filename with a timestamp."""
    timestamp = generate_timestamp(timestamp_format)
    return f"{prefix}_{timestamp}.pth"

def parse_mnist_image(raw_bytes):
    # 解析 IDX 文件中的图像数据
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    images = data[16:].reshape(-1, 28, 28)  # 跳过前 16 字节，并重塑为图像格式
    return images

# 解析 MNIST 标签数据
def parse_mnist_labels(raw_bytes):
    # 解析 IDX 文件中的标签数据
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    labels = data[8:]  # 跳过前 8 字节，剩下的是标签
    return labels

def print_image_by_column(image):
    """按列打印图像（28x28），每行打印28个像素值，确保列宽为3个字符"""
    for row in range(image.shape[0]):  # 对于每一行
        for col in range(image.shape[1]):  # 对于每一列
            # 使用格式化字符串来控制每列宽度为3个字符
            print(f"{image[row, col]:3d}", end=" ")  # 打印每个像素值，宽度为3
        print()  # 每行之后换行
    print("\n")  # 添加额外换行，用于分隔不同的图像

# 假设 train_images 和 train_labels 是存储 MNIST 图像和标签字节数据的 RDDs
def get_mnist_data(images_rdd, labels_rdd, is_train):
    # 使用 flatMap 来将每个文件中的多个图像展开
    images_rdd = images_rdd.flatMap(lambda x: parse_mnist_image(x[1]))  # 解析图像数据
    labels_rdd = labels_rdd.flatMap(lambda x: parse_mnist_labels(x[1]))  # 解析标签数据

    if is_train: 
        #print(images_rdd.take(1)[0])  # 输出前 1 个元素查看格式
        print_image_by_column(images_rdd.take(1)[0])  # 按列打印图像
        print(labels_rdd.take(1)[0])  # 输出前 1 个元素查看格式


    # 将图像和标签配对
    labeled_image_rdd = images_rdd.zip(labels_rdd)

    # 使用 reduceByKey 来统计每个标签的出现次数
    label_counts = labeled_image_rdd.map(lambda x: (x[1], 1))  # (标签, 1)
    label_count_rdd = label_counts.reduceByKey(lambda a, b: a + b)

    if is_train:
        # Collect the result to print it
        label_count_result = label_count_rdd.collect()
        for label, count in label_count_result:
            print(f"Label {label}: {count} times")

    return labeled_image_rdd  # 返回配对的数据

def prepare_data(train_images_path, train_labels_path, t10k_images_path, t10k_labels_path, sparkSession):
    """
    Prepare the training and testing datasets from binary files.
    """
    train_images = sparkSession.binaryFiles(train_images_path)
    train_labels = sparkSession.binaryFiles(train_labels_path)
    test_images = sparkSession.binaryFiles(t10k_images_path)
    test_labels = sparkSession.binaryFiles(t10k_labels_path)

    train_data_rdd = get_mnist_data(train_images, train_labels, is_train=True)
    test_data_rdd = get_mnist_data(test_images, test_labels, is_train=False)

    train_data = train_data_rdd.collect()
    test_data = test_data_rdd.collect()

    return train_data, test_data


def build_datasets(train_data, test_data):
    """
    Build PyTorch datasets from the processed MNIST data.
    """
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)

    train_dataset = CT.MNISTDataset(train_images, train_labels)
    test_dataset = CT.MNISTDataset(test_images, test_labels)

    return train_dataset, test_dataset

def train(learning_rate=0.01, use_gpu=True, train_dataset=None):
    import torch.nn as nn
    import torch.optim as optim

    dist.init_process_group(backend='nccl')

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if use_gpu else torch.device("cpu")
    
    model = CT.ImprovedCNN2().to(device)
    model = DDP(model, device_ids=[device] if use_gpu else None)
    
    sampler = DistributedSampler(train_dataset)
    loader = DataLoader(train_dataset, sampler=sampler, batch_size=8)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(10):  # Example epoch count
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1} completed, loss: {loss.item()}")
    dist.destroy_process_group()
    return model
        # Clear CUDA cache


train_data, test_data = prepare_data(
        mnist_paths["train_images"],
        mnist_paths["train_labels"],
        mnist_paths["test_images"],
        mnist_paths["test_labels"],
        sc
    )

train_dataset, test_dataset = build_datasets(train_data, test_data)
distributor = TorchDistributor(num_processes=1, local_mode=True, use_gpu=True)
distributor.run(train, learning_rate=0.01, use_gpu=True, train_dataset=train_dataset)
sc.stop()
#distributor = pyspark.ml.torch.killdistributor.TorchDistributor()
#distributor = sc.ml.torch.distributor.TorchDistributor()