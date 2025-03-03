#/bin/bash

kubectl cp master:/config/Codebase ../Codebase

kubectl delete -f service.yaml -f pvc.yaml -f master-pod.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml