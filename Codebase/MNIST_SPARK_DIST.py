import os
import time
import subprocess
import logging
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
import CNN_TOOLS as CT
import numpy as np
import pyspark
import gc

os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
print("SparkContext initialized.")
print(f"Application ID: {sc.applicationId}")

# Training Configuration
EPOCH_COUNT = 15         # Number of training epochs
BATCH_SIZE = 64          # Number of samples per batch
useGpu = True            # DO NOT CHANGE FROM TRUE

hdfs_base_path = "hdfs://master:9000/user/MNIST"
local_checkpoint_file_path = "/config/model_checkpoint.pth"
hdfs_checkpoint_dir = "hdfs://master:9000/user/MNIST_Checkpoint"


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
    images_rdd = images_rdd.flatMap(lambda x: parse_mnist_image(x[1]))  #> 解析图像数据
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
    gc.collect()
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

def load_checkpoint(model, optimizer, filename="/config/model_checkpoint.pth"):
    """
    Load model checkpoint if it exists. If the file does not exist, return a starting epoch and loss of 0.
    """
    epoch = 0
    loss = 0.0
    device = torch.device(int(os.environ["LOCAL_RANK"]) if useGpu else "cpu")
    # Only rank 0 should load the checkpoint
    if os.path.exists(filename):
        checkpoint = torch.load(filename, weights_only=True)
        print(f"Loaded checkpoint from {filename}")
        #epoch = checkpoint['epoch']
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint.get('loss', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from epoch {epoch - 1}, loss {loss}")

        epoch_tensor = torch.tensor(epoch).to(device)
        loss_tensor = torch.tensor(loss).to(device)
        # Broadcast the epoch and loss to all ranks
    
        return epoch_tensor.item(), loss_tensor.item()
    else:
        print("Checkpoint not found or rank != 0, starting from scratch.")
        # Skip broadcasting and return default values when no checkpoint is found
        epoch_tensor = torch.tensor(epoch).to(device)
        loss_tensor = torch.tensor(loss).to(device)

        # No broadcast needed, as all ranks will start from scratch
        # Only return the default epoch and loss without broadcasting
        return epoch_tensor.item(), loss_tensor.item()


def save_checkpoint(model, optimizer, epoch, loss, filename="/config/model_checkpoint.pth"):
    """
    Save the checkpoint with model state and optimizer state.
    Only rank 0 saves the checkpoint.
    保存模型和优化器的状态： 在训练过程中，通常会在每个 epoch 或某个间隔时保存模型的状态和优化器的状态。保存的信息通常包括：

    模型的参数 (model.state_dict())
    优化器的状态 (optimizer.state_dict())
    当前的 epoch 和 loss 等其他训练状态
    """
    if torch.distributed.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename} at epoch {epoch}")


def upload_checkpoint_to_hdfs(local_checkpoint_file, hdfs_checkpoint_dir):
    # Check if the local checkpoint file exists
    if not os.path.exists('/config/'+ local_checkpoint_file):
        print(f"Local file {local_checkpoint_file} does not exist, unable to upload.")
        return

    try:
        # Ensure the HDFS directory exists
        sc.addFile("/config/" + local_checkpoint_file)
        hdfs_file_path_timestamp= os.path.join(hdfs_checkpoint_dir, local_checkpoint_file)
        with open(SparkFiles.get("/config/" + local_checkpoint_file), "rb") as f:
            file_content = f.read()
        spark.sparkContext.parallelize([file_content]).saveAsTextFile(hdfs_file_path_timestamp)
        os.remove("/config/" + local_checkpoint_file)
    except subprocess.CalledProcessError as e:
        print(f"HDFS operation failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on the given dataset.
    The function evaluates the model's performance by comparing the predicted labels
    with the true labels for all batches in the data loader.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through the dataset in the data loader
        for data, target in data_loader:
            # Get model predictions
            output = model(data)

            # Get the predicted class with the highest probability
            _, predicted = torch.max(output, 1)

            # Count the number of correct predictions
            correct += (predicted == target).sum().item()
            total += target.size(0)

    # Calculate the accuracy as a percentage
    accuracy = 100 * correct / total

    return accuracy


def show_sample_images(dataset, num_images=10):
    """
    Display a specified number of sample images from the dataset.
    The function randomly selects images and labels from the dataset and plots them.
    """
    # Check if the dataset is empty
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Ensure that the number of images to display does not exceed the dataset size
    num_images = min(num_images, len(dataset))

    # Randomly sample indices from the dataset
    sample_indices = random.sample(range(len(dataset)), num_images)
    images, labels = zip(*[dataset[i] for i in sample_indices])

    # Plot the sampled images
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, num_images, i + 1)

        # Ensure the image is 2D by removing the channel dimension (if any)
        image = image.squeeze(0)  # Remove channel dimension (1, 28, 28) -> (28, 28)

        # Display the image in grayscale
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")

    # Show the plot
    plt.show()


def plot_loss_curve(loss_dict):
    """
    Plot the loss curve for each rank during training.
    The function visualizes the loss over batches for each rank in a distributed training setting.
    """
    plt.figure(figsize=(12, 6))

    # Loop through each rank and plot its loss curve
    for rank, rank_loss in loss_dict:
        plt.plot(rank_loss, label=f"Rank {rank}")

    # Add labels and title to the plot
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss Curve for Each Rank")
    plt.legend()

    # Show the plot
    plt.show()


def plot_accuracy_curve(accuracy_dict):
    """
    Plot the training accuracy curve for each rank.
    The function visualizes the accuracy over batches for each rank in a distributed training setting.
    """
    plt.figure(figsize=(10, 6))

    # Loop through each rank and plot its accuracy curve
    for rank, accuracy in accuracy_dict:
        plt.plot(accuracy, label=f'Rank {rank}')

    # Add labels, title, and grid to the plot
    plt.title("Training Accuracy for Each Rank")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_confusion_matrix(model, data_loader, num_classes=10):
    """
    Plot the confusion matrix for the model's predictions.
    The function evaluates the model on the data_loader and visualizes the confusion matrix using seaborn.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        # Iterate through the data_loader
        for data, target in data_loader:
            # Get the model's predictions
            output = model(data)
            _, predicted = torch.max(output, 1)

            # Store the predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model, test_dataset):
    from torch.utils.data import DataLoader
    """
    Evaluate the model on the test dataset and display the results.
    """
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Calculate test accuracy
    final_accuracy = calculate_accuracy(model, test_loader)

    # Create a visually framed output for the test results
    print("=" * 53)
    print(f"|{'Final Test Results':^51}|")
    print("=" * 53)
    print(f"|{'Test Accuracy:':<30} {final_accuracy:.2f}%{' ':<14}|")
    print("=" * 53)

    # Plot and display the confusion matrix
    # plot_confusion_matrix(model, test_loader)


def train(initializer, use_gpu=True, train_dataset=None, restart=True):
    import os
    import subprocess
    import torch
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    from pyspark.ml.torch.distributor import TorchDistributor
    import CNN_TOOLS as CT
    import numpy as np
    import pyspark
    import logging
    import gc
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
 
    log_filename = f"rank_{int(os.environ['RANK'])}_training_log.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    gc.collect()
    if use_gpu: torch.cuda.empty_cache()  # Clear GPU memory

    # Initialize the process group
    os.environ['NCCL_DEBUG'] = 'ERROR'
    dist.init_process_group(backend='nccl' if use_gpu else 'gloo')

    manager = torch.multiprocessing.Manager()

    loss_dict = manager.dict()
    accuracy_dict = manager.dict()
    batch_loss_dict = manager.dict()

    # Set the device for this process
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    model, optimizer, _  = initializer(device)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if restart:
        # Since the checkpoint is stored in the shared config directory, all ranks can access it
        start_epoch, loss = load_checkpoint(model, optimizer, filename="/config/model_checkpoint.pth")
    else:
        start_epoch = 0
        loss = 0.0

    sampler = DistributedSampler(train_dataset)
    loader = DataLoader(train_dataset, sampler=sampler, batch_size=BATCH_SIZE)
    
    dist.barrier()

    criterion = nn.CrossEntropyLoss()

    sampler.set_epoch(start_epoch)  # Set the correct epoch

    # Record the start time for each worker
    start_time = time.time()  # Record start time

    # Initialize variables to track total loss
    total_loss = 0.0
    total_batches = 0

    # Record loss for each rank
    rank_loss = []  # This will store the loss for each batch in this rank
    rank_accuracy = []  # List to store accuracy for each batch in this rank

    model.train()
    for epoch in range(start_epoch, EPOCH_COUNT):  # Example epoch count
        sampler.set_epoch(epoch)  # Ensure proper shuffling
        epoch_loss = 0
        correct = 0
        total = 0
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            total_loss += loss.item()  # Accumulate total loss
            total_batches += 1  # Count total batches

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Record the loss for each batch
            rank_loss.append(loss.item())  # Append loss of this batch to rank's list
            accuracy = 100 * correct / total
            rank_accuracy.append(accuracy)
        
        epoch_accuracy = 100 * correct / total
        rank_accuracy.append(epoch_accuracy)

        dist.barrier()

          # Store the loss and accuracy for the rank
        batch_loss_dict[global_rank] = rank_loss  # Store batch-wise losses for this rank
        loss_dict[global_rank] = epoch_loss / len(loader)  # Store average loss for the rank
        accuracy_dict[global_rank] = rank_accuracy  # Store accuracy for each rank

        if global_rank == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), filename="/config/model_checkpoint.pth")
        
        gc.collect()
        if use_gpu: torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} on device {global_rank} completed, loss: {loss.item()}")

    end_time = time.time()  # Record end time
    training_time = end_time - start_time  # Calculate total training time

    dist.barrier()

    # Aggregate loss across all ranks
    total_loss = torch.tensor(total_loss).to(device)
    dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)

    # Aggregate accuracy across all ranks
    total_accuracy = torch.tensor(epoch_accuracy).to(device)
    dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, dst=0)

    if global_rank == 0:
        total_accuracy /= world_size
        avg_loss = total_loss.item() / total_batches  # Average loss across all batches

        # Format and print training results
        print(f"\n{'='*53}")
        if world_size > 1:
            print(f"Training Results: Parallel Training - World Size: {world_size}")
        else:
            print("Training Results: Single Node Training")
        print(f"{'='*53}")
        print(f"Total Epochs: {EPOCH_COUNT}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Final Training Loss:       {avg_loss:.4f}")  # Report average loss
        print(f"Final Training Accuracy:   {total_accuracy.item():.2f}%")
        print(f"Total Training Time:       {training_time:.2f} seconds")
        print(f"{'='*53}")

    dist.barrier()

    # Destroy the process group
    dist.destroy_process_group()
    model = model.to("cpu")
    return model.module, loss_dict.items(), accuracy_dict.items(), batch_loss_dict.items()

# Prepare data
print("Building datasets...", end=" ")

start = time.time()

train_data, test_data = prepare_data(
    mnist_paths["train_images"],
    mnist_paths["train_labels"],
    mnist_paths["test_images"],
    mnist_paths["test_labels"],
    sc
)

train_dataset, test_dataset = build_datasets(train_data, test_data)
print(f"Done. Elapsed time: {time.time() - start} seconds.")

# Use TorchDistributor to run the training function
os.environ["NCCL_DEBUG"] = "ERROR"

print("Running TorchDistributor...", end=" ")

start = time.time()

distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=useGpu)

cnn_model, loss_dict, accuracy_dict, batch_loss_dict = distributor.run(
    train,
    CT.initialize_cnn,
    use_gpu=useGpu,
    train_dataset=train_dataset,
    restart=False
)

print(f"Exited Distributed Process. Elapsed time: {time.time() - start} seconds.")


filename = generate_checkpoint_filename(prefix="CNN")

os.rename("/config/model_checkpoint.pth", "/config/" + filename)

upload_checkpoint_to_hdfs(filename, hdfs_checkpoint_dir)


plot_loss_curve(batch_loss_dict)  # Plot loss curve
plot_accuracy_curve(accuracy_dict)  # Plot accuracy curve
evaluate_model(cnn_model, test_dataset)  # Model evaluation


start = time.time()

res18_model, loss_dict, accuracy_dict, batch_loss_dict = distributor.run(
    train,
    CT.initialize_res18,
    use_gpu=useGpu,
    train_dataset=train_dataset,
    restart=False
)

print(f"Exited Distributed Process. Elapsed time: {time.time() - start} seconds.")

filename = generate_checkpoint_filename(prefix="Resnet_18")

os.rename("/config/model_checkpoint.pth", "/config/" + filename)

upload_checkpoint_to_hdfs(filename, hdfs_checkpoint_dir)

plot_loss_curve(batch_loss_dict)  # Plot loss curve
plot_accuracy_curve(accuracy_dict)  # Plot accuracy curve
evaluate_model(res18_model, test_dataset)  # Model evaluation

start = time.time()

res50_model, loss_dict, accuracy_dict, batch_loss_dict = distributor.run(
    train,
    CT.initialize_res50,
    use_gpu=useGpu,
    train_dataset=train_dataset,
    restart=False
)

print(f"Exited Distributed Process. Elapsed time: {time.time() - start} seconds.")

filename = generate_checkpoint_filename(prefix="Resnet50")

os.rename("/config/model_checkpoint.pth", "/config/" + filename)

upload_checkpoint_to_hdfs(filename, hdfs_checkpoint_dir)

plot_loss_curve(batch_loss_dict)  # Plot loss curve
plot_accuracy_curve(accuracy_dict)  # Plot accuracy curve
evaluate_model(res50_model, test_dataset)  # Model evaluation

# Stop the Spark context
sc.stop()