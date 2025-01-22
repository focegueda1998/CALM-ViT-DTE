import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import io
from PIL import Image
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession, Row
from pyspark.mllib.linalg import Matrices, DenseMatrix
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StructField, StructType, BinaryType, StringType
from pyspark.ml.image import ImageSchema

# Function to unpickle the batch files

def unpickle(batch_path):
    batch = None
    with open(SparkFiles.get(batch_path), 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    return batch

# Function to add pickle file to SparkFiles

def addFileToSpark(batch_path):
    sc.addFile(pickle_path + batch_path)

def imageToBinary(image):
    with io.BytesIO() as stream:
        image.save(stream, format='JPEG')
        return stream.getvalue()
    return None

# Spark configuration

hdfs_path = "hdfs://master:9000/user/"
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
print(f"Spark session created! Application ID: {sc.applicationId}")

# Get the paths of the batches, then parallelize them

pickle_path = (hdfs_path + "cifar-10-batches-py/")

batch_paths = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']

for batch_path in batch_paths:
    addFileToSpark(batch_path)

# Merge the data and labels into a single RDD

rdd_batch = sc.parallelize(batch_paths) \
            .map(unpickle) \
            .map(lambda x: (x[b'data'], x[b'labels'])) \
            .flatMap(lambda x: [(x[0][i], x[1][i]) for i in range(len(x[0]))]) \
            .map(lambda x: (torch.tensor(x[0]).cuda().reshape(3, 32, 32).permute(1, 2, 0).cpu(), x[1]))

schema = StructType([
    StructField("image", BinaryType()),
    StructField("label", IntegerType()),
    StructField("filename", StringType())
])

# Save images to HDFS

sc.addFile(pickle_path + 'batches.meta')
meta = unpickle('batches.meta')
label_names = [label.decode('utf-8') for label in meta[b'label_names']]

df = rdd_batch.map(lambda x: (Image.fromarray(x[0]), x[1])) \
    .map(lambda x: (imageToBinary(x[0]), x[1], f"{label_names[x[1]]}")) \
    .toDF(schema)


for label, images in rdd_batch.groupBy(lambda x: x[1]).collect():
    label_names = [label.decode('utf-8') for label in meta[b'label_names']]
    image = sc.parallelize(images).map(lambda x: x[0]).collect()
    plt.imshow(image[random.randint(0, 6000)], interpolation='bicubic')
    plt.xlabel(label_names[label])
    plt.show()
    print(f"Label: {label_names[label]}, Number of images: {len(images)}")

sc.stop()