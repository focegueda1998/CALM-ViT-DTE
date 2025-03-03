#/bin/bash

echo "Copying Codebase from shared volume..."
kubectl cp master:config/Codebase/ ../Codebase/

echo "Deleting deployment..."
kubectl delete -f service.yaml -f pvc.yaml -f master-pod.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml