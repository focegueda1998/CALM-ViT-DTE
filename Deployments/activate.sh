#!/bin/bash

kubectl apply -f ingress.yaml -f service.yaml -f pvc.yaml -f master-pod.yaml -f slave01-pod.yaml -f slave02-pod.yaml