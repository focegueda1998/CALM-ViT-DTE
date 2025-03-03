#!/bin/bash

echo "Deploying Spark Cluster..."
kubectl apply -f ingress.yaml -f service.yaml -f pvc.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml -f master-pod.yaml

while kubectl get pods | grep -q -E "Pending|Init:[0-9]+/[0-9]+|ContainerCreating|Succeeded|Failed|Unknown|CrashLoopBackOff|Error";
do
    echo "Waiting for pods to be ready..."
    sleep 5
done

echo "Copying Codebase to shared volume..."
kubectl cp ../Codebase master:/config
echo "Codebase copied to shared volume. Deployment complete."
echo "Waiting 20700 seconds until deactivating..."
sleep 20700
echo "Executing deactivation script..."
bash ./deactivate.sh