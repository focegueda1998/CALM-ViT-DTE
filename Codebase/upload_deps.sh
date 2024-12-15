#!/bin/bash

zip py_files.zip CNN_TOOLS.py MNIST_SPARK_DIST.py;
hadoop fs -rm -r /user/Codebase;
hadoop fs -rm -r /user/MNIST;
hadoop fs -mkdir /user/Codebase;
hadoop fs -mkdir /user/MNIST;
hadoop fs -put py_files.zip MNIST_SPARK_DIST.py /user/Codebase;
hadoop fs -put MNIST /user/;