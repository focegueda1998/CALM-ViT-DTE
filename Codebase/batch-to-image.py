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


def secondsToTime(seconds):
    secs = int(seconds) % 60
    mins = int(seconds // 60) % 60
    hours = int(mins // 60)
    return f"{hours if hours > 9 else f'0{hours}'}:{mins if mins > 9 else f'0{mins}'}:{secs if secs > 9 else f'0{secs}'}"

# Spark configuration
start = time()

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
    image = sc.parallelize(images).map(lambda x: (x[1], x[2])).collect()
    index = random.randint(0, len(image) - 1)
    # showImage(image[index][1], image[index][0])
    with open(f"/home/Codebase/images/{image[index][0]}.jpg", "wb") as file:
        file.write(image[index][1])
    print(f"Label: {label_names[label]}, Number of images: {len(images)}")

rdds = []

for label in ['dog', 'cat', 'wild']:
    rdds.append(
        spark.read.format("binaryFile") \
             .option("pathGlobFilter", "*.png") \
             .option("recursiveFileLookup", False) \
             .load(f"/user/animals/{label}") \
             .rdd.map(
                 lambda x: (
                     label_names.index(label), 
                     f"{label}_512x512_{str(uuid.uuid4())}", 
                     x["content"],
                     random.random()
                 ) 
              )
             .sortBy(lambda x: x[3]) \
             .map(lambda x: (x[0], x[1], x[2]))
    )

    rdds[-1].toDF(Schema).write.mode("overwrite").parquet(f"/user/animal-images/{label}")

    animal = spark.read.parquet("/user/animal-images/" + label).rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))

    # showImage(animal.first()[2], animal.first()[1])
   
    with open(f"/home/Codebase/images/{animal.first()[1]}.png", "wb") as file:
        file.write(animal.first()[2])
    print(f"Label: {label_names[animal.first()[0]]}, Number of images: {animal.count()}")

dog_rdds = rdds[0].randomSplit([0.2 for _ in range(5)], random.randint(0, 100))
cat_rdds = rdds[1].randomSplit([0.2 for _ in range(5)], random.randint(0, 100))
wld_rdds = rdds[2].randomSplit([0.2 for _ in range(5)], random.randint(0, 100))

for i in range(5):

    dog_rdds[i].union(cat_rdds[i]) \
               .union(wld_rdds[i]) \
               .map(lambda x: (x[0], x[1], x[2], random.random())) \
               .sortBy(lambda x: x[3]) \
               .map(lambda x: (x[0], x[1], x[2])) \
               .toDF(Schema).repartition(12) \
               .write.mode("overwrite").parquet(f"/user/train/batch_{i}")
    
    animal = spark.read.parquet(f"/user/train/batch_{i}").rdd.map(lambda x: (x["label"], x["fileID"], x["data"]))

    # showImage(animal.first()[2], animal.first()[1])

    with open(f"/home/Codebase/images/{animal.first()[1]}.png", "wb") as file:
        file.write(animal.first()[2])
    print(f"Label: {label_names[animal.first()[0]]}, Number of images: {animal.count()}")

print(f"Time taken: {secondsToTime(time() - start)}")

sc.stop()