#!/bin/bash

kubectl apply -f ingress.yaml -f service.yaml -f pvc.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml -f master-pod.yaml