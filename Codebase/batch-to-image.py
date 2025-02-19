import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import io
import uuid
from PIL import Image
from time import time
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, udf, rand
from pyspark.sql.types import IntegerType, StructField, StructType, BinaryType, StringType, ArrayType
from subprocess import call

# Create the images directory if it doesn't exist

call(['mkdir', '-p', '/home/Codebase/images'])

#? Function to unpickle a batch file
def unpickle(batch_path):
    batch = None
    with open(SparkFiles.get(batch_path), 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    return batch

#? Function to add pickle file to SparkFiles

def addFileToSpark(pickle_path, batch_path):
    sc.addFile(pickle_path + batch_path)

#? Function to save an image to HDFS (Read the note below)
#! While this technically works, bulk datasets should be saved in parquet files
#! to avoid the overhead of saving each image individually. If it is necessary to
#! save them indvidually, be aware each image will be stored as a separate block.
#! By default, HDFS has a block size of 128MB, so each image will take up that much
#! space despite its size.
def saveImageToHDFS(image, path, fsys):
    out = fsys.create(path)
    out.write(image)
    out.close()

#? Function to convert an image to binary
def imageToBinary(image):
    with io.BytesIO() as stream:
        image.save(stream, format='JPEG', quality=100, keep_rgb=True)
        return stream.getvalue()
    return None

#? Function to show an image using matplotlib
def showImage(image, label):
    plt.imshow(
        np.array(
            Image.open(
                io.BytesIO(
                    image
                )
            )
        ), 
        interpolation='bicubic'
    )
    plt.title(label)
    plt.show()

#? Function to convert seconds to a something more readable.
def secondsToTime(seconds):
    secs = int(seconds) % 60
    mins = int(seconds // 60) % 60
    hours = int(mins // 60)
    return f"{hours if hours > 9 else f'0{hours}'}:{mins if mins > 9 else f'0{mins}'}:{secs if secs > 9 else f'0{secs}'}"

# Start the timer
start = time()

# Create the Spark session
hdfs_path = "hdfs://master:9000/user/"
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# List the contents of the user directory
gateway = sc._gateway
FileSystem = gateway.jvm.org.apache.hadoop.fs.FileSystem
Configuration = gateway.jvm.org.apache.hadoop.conf.Configuration
Path = gateway.jvm.org.apache.hadoop.fs.Path
URI = gateway.jvm.java.net.URI
conf = Configuration()
fs = FileSystem.get(URI("hdfs://master:9000/"), conf)
status = fs.listStatus(Path("/user/"))
for file in status:
    print(file.getPath().toString())

# Print the application ID
print(f"Spark session created! Application ID: {sc.applicationId}")

# Define the schema for the image data, this will store the label, fileID and the image data
Schema = StructType([
    StructField("label", IntegerType()),
    StructField("fileID", StringType()),
    StructField("data", BinaryType())
])

# Get the paths of the batches, then parallelize them
metadeta_df = spark.read.csv("file:///config/AI_Human_Generated_Images/train.csv", header=True, inferSchema=True) \
                   .withColumn("file_name", regexp_replace("file_name",".*train_data/", "")).repartition(1)
metadeta_df.show()
r_or_f_df  = spark.read.format("binaryFile") \
                  .option("pathGlobFilter", "*.jpg") \
                  .option("recursiveFileLookup", False) \
                  .load(f"file:///config/AI_Human_Generated_Images/train_data/") \
                  .repartition(40) \
                  .withColumnRenamed("path", "file_name") \
                  .withColumn("file_name", regexp_replace("file_name", ".*train_data/", "")) \
                  .join(metadeta_df, "file_name") \
                  .select("label", "file_name", "content") \
                  .withColumnRenamed("content", "data") \
                  .withColumnRenamed("file_name", "fileID")
r_or_f_df.show()
dfs = r_or_f_df.randomSplit([0.1 for _ in range(10)], random.randint(0, 100))

for i in range(10):
    dfs[i] = dfs[i].repartition(4).select("*").orderBy(rand())
    dfs[i].write.mode("overwrite").parquet(f"/user/real_or_fake/batch_{i}")
    dfs[i].show()
    r_or_f = spark.read.parquet(f"/user/real_or_fake/batch_{i}").rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))
    # showImage(r_or_f.first()[2], r_or_f.first()[1])
    with open(f"/home/Codebase/images/{r_or_f.first()[1]}.jpg", "wb") as file:
        file.write(r_or_f.first()[2])
    print(f"Label: {r_or_f.first()[0]}, Number of images: {r_or_f.count()}")

print(f"Time taken: {secondsToTime(time() - start)}")

# pickle_path = (hdfs_path + "cifar-10-batches-py/")
# batch_paths = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']
# for batch_path in batch_paths:
#     addFileToSpark(pickle_path, batch_path)

# # Merge the data and labels into a single RDD
# sc.addFile(pickle_path + 'batches.meta')
# meta = unpickle('batches.meta')
# label_names = [label.decode('utf-8') for label in meta[b'label_names']] + ['wild']

# # Define the schema for the image data, this will store the label, fileID and the image data
# Schema = StructType([
#     StructField("label", IntegerType()),
#     StructField("fileID", StringType()),
#     StructField("data", BinaryType())
# ])

# # Read the batch files, unpickle them, then convert the data to a format that can be saved as an image
# sc.parallelize(batch_paths) \
#   .map(unpickle) \
#   .map(lambda x: (x[b'data'], x[b'labels'])) \
#   .flatMap(lambda x: [(x[0][i], x[1][i]) for i in range(len(x[0]))]) \
#   .map(lambda x: (torch.tensor(x[0]).cuda().reshape(3, 32, 32).permute(1, 2, 0).cpu().numpy(), x[1])) \
#   .map(lambda x: (x[1], f"{label_names[x[1]]}_32x32_{str(uuid.uuid4())}", Image.fromarray(x[0].astype(np.uint8)))) \
#   .map(lambda x: (x[0], x[1], imageToBinary(x[2]))).toDF(Schema).repartition(2) \
#   .write.mode("overwrite").parquet("/user/cifar-10-images")

# # Read the cifar-10-images from HDFS, group them by label, then save one image from each label to the images directory
# rdd_batch = spark.read.parquet("/user/cifar-10-images").rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))
# for label, images in rdd_batch.groupBy(lambda x: x[0]).collect():
#     image = sc.parallelize(images).map(lambda x: (x[1], x[2])).collect()
#     index = random.randint(0, len(image) - 1)
#     # showImage(image[index][1], image[index][0])
#     with open(f"/home/Codebase/images/{image[index][0]}.jpg", "wb") as file:
#         file.write(image[index][1])
#     print(f"Label: {label_names[label]}, Number of images: {len(images)}")

# # Read the animal images from HDFS, group them by label, then save one image from each label to the images directory
# rdds = []
# for label in ['dog', 'cat', 'wild']:
#     rdds.append(
#         spark.read.format("binaryFile") \
#              .option("pathGlobFilter", "*.png") \
#              .option("recursiveFileLookup", False) \
#              .load(f"/user/animals/{label}") \
#              .rdd.map(
#                  lambda x: (
#                      label_names.index(label), 
#                      f"{label}_512x512_{str(uuid.uuid4())}", 
#                      x["content"],
#                      random.random()
#                  ) 
#               ) 
#              .sortBy(lambda x: x[3]) \
#              .map(lambda x: (x[0], x[1], x[2]))
#     )
#     rdds[-1].toDF(Schema).write.mode("overwrite").parquet(f"/user/animal-images/{label}")
#     animal = spark.read.parquet("/user/animal-images/" + label).rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))
#     # showImage(animal.first()[2], animal.first()[1])
#     with open(f"/home/Codebase/images/{animal.first()[1]}.png", "wb") as file:
#         file.write(animal.first()[2])
#     print(f"Label: {label_names[animal.first()[0]]}, Number of images: {animal.count()}")

# # Split the animal images into 5 batches, then union them together
# dog_rdds = rdds[0].randomSplit([0.2 for _ in range(5)], random.randint(0, 100))
# cat_rdds = rdds[1].randomSplit([0.2 for _ in range(5)], random.randint(0, 100))
# wld_rdds = rdds[2].randomSplit([0.2 for _ in range(5)], random.randint(0, 100))
# for i in range(5):
#     dog_rdds[i].union(cat_rdds[i]) \
#                .union(wld_rdds[i]) \
#                .map(lambda x: (x[0], x[1], x[2], random.random())) \
#                .sortBy(lambda x: x[3]) \
#                .map(lambda x: (x[0], x[1], x[2])) \
#                .toDF(Schema).repartition(12) \
#                .write.mode("overwrite").parquet(f"/user/train/batch_{i}")
#     animal = spark.read.parquet(f"/user/train/batch_{i}").rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))
#     # showImage(animal.first()[2], animal.first()[1])
#     with open(f"/home/Codebase/images/{animal.first()[1]}.png", "wb") as file:
#         file.write(animal.first()[2])
#     print(f"Label: {label_names[animal.first()[0]]}, Number of images: {animal.count()}")

# # Print the time taken, then stop the Spark session
print(f"Time taken: {secondsToTime(time() - start)}")
sc.stop()