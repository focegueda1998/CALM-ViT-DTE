import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import io
import uuid
from PIL import Image
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType, BinaryType, StringType

# Function to unpickle the batch files

def unpickle(batch_path):
    batch = None
    with open(SparkFiles.get(batch_path), 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    return batch

# Function to add pickle file to SparkFiles

def addFileToSpark(batch_path):
    sc.addFile(pickle_path + batch_path)

def saveImageToHDFS(image, path, fsys):
    out = fsys.create(path)
    out.write(image)
    out.close()

def imageToBinary(image):
    with io.BytesIO() as stream:
        image.save(stream, format='JPEG', quality=100, keep_rgb=True)
        return stream.getvalue()
    return None

# Spark configuration

hdfs_path = "hdfs://master:9000/user/"
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
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
print(f"Spark session created! Application ID: {sc.applicationId}")

# Get the paths of the batches, then parallelize them

pickle_path = (hdfs_path + "cifar-10-batches-py/")

batch_paths = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']

for batch_path in batch_paths:
    addFileToSpark(batch_path)

# Merge the data and labels into a single RDD

sc.addFile(pickle_path + 'batches.meta')
meta = unpickle('batches.meta')

label_names = [label.decode('utf-8') for label in meta[b'label_names']] + ['wild']

Schema = StructType([
    StructField("label", IntegerType()),
    StructField("fileID", StringType()),
    StructField("data", BinaryType())
])

sc.parallelize(batch_paths) \
  .map(unpickle) \
  .map(lambda x: (x[b'data'], x[b'labels'])) \
  .flatMap(lambda x: [(x[0][i], x[1][i]) for i in range(len(x[0]))]) \
  .map(lambda x: (torch.tensor(x[0]).cuda().reshape(3, 32, 32).permute(1, 2, 0).cpu().numpy(), x[1])) \
  .map(lambda x: (x[1], f"{label_names[x[1]]}_32x32_{str(uuid.uuid4())}", Image.fromarray(x[0].astype(np.uint8)))) \
  .map(lambda x: (x[0], x[1], imageToBinary(x[2]))).toDF(Schema).repartition(2) \
  .write.mode("overwrite").parquet("/user/cifar-10-images")


rdd_batch = spark.read.parquet("/user/cifar-10-images").rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))

for label, images in rdd_batch.groupBy(lambda x: x[0]).collect():
    # image = sc.parallelize(images).map(lambda x: (x[1], x[2])).collect()
    # index = random.randint(0, len(image) - 1)
    # plt.imshow(
    #     np.array(
    #         Image.open(
    #             io.BytesIO(
    #                 image[index][1]
    #             )
    #         )
    #     ), 
    #     interpolation='bicubic'
    # )
    # plt.xlabel(image[index][0])
    # plt.show()
    print(f"Label: {label_names[label]}, Number of images: {len(images)}")

for label in ['dog', 'cat', 'wild']:
    spark.read.format("binaryFile") \
         .option("pathGlobFilter", "*.png") \
         .option("recursiveFileLookup", False) \
         .load(f"/user/animals/{label}") \
         .rdd.map(lambda x: (label_names.index(label), f"{label}_512x512_{str(uuid.uuid4())}", x["content"])) \
         .toDF(Schema).write.mode("overwrite").parquet("/user/animal-images/" + label)
    
    animal = spark.read.parquet("/user/animal-images/" + label).rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))

    plt.imshow(
        np.array(
            Image.open(
                io.BytesIO(
                    animal.first()[2]
                )
            )
        ), 
        interpolation='bicubic'
    )
    plt.xlabel(animal.first()[1])
    plt.show()
    print(f"Label: {label_names[animal.first()[0]]}, Number of images: {animal.count()}")

sc.stop()