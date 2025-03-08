#!/bin/bash

SECONDS=0
KILL_AFTER_TRAINING=true

echo "Deploying Spark Cluster..."
kubectl apply -f ingress.yaml -f service.yaml -f pvc.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml -f master-pod.yaml
echo "Waiting for pods to be ready..."

while kubectl get pods | grep -q -E "Pending|Init:[0-9]+/[0-9]+|ContainerCreating|Succeeded|Failed|Unknown|CrashLoopBackOff|Error"; do
    sleep 5
done

clear
echo "All pods are ready... Let's take a breather..."
sleep 30

clear
echo "Copying Codebase to shared volume..."
kubectl cp --retries=3 ../Codebase master:/config
echo "Codebase copied to shared volume. Deployment complete."
echo "Waiting for traing to end or 21000 seconds until deactivating..."
export kill=false
until [ $SECONDS -ge 21000 ] || [ "$kill" = "true" ]; do
    if [ "$KILL_AFTER_TRAINING"  = "true" ]; then
        export tail=$(kubectl logs --tail=8 master)
        echo "Inquiring master, last 4 lines:"
        echo "$tail"
        if echo "$tail" | grep -q "TRAINING HAS ENDED"; then
            export kill=true
            echo "Training ended."
        fi
    fi
    export remaining=$(((21000 - $SECONDS) / 60))
    echo "Time remaining: $remaining minutes"
    sleep 60
    clear
done

if [ "$kill" = "false" ]; then
    echo "Training did not end in 21000 seconds. Deactivating..."
elif [ "$kill" = "true" ]; then
    kubectl logs --tail=4 master
    echo "Training ended. Deactivating..."
fi

echo "Executing deactivation script..."
bash ./deactivate.sh