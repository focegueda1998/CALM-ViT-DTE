#/bin/bash

echo "Copying Codebase from shared volume..."
export BACKUP_DIR="../Codebase-$(date +'%Y-%m-%d-%H-%M-%S')"
mkdir -p $BACKUP_DIR
mv -T "../Codebase" $BACKUP_DIR
kubectl cp --retries=3 master:config/Codebase/ "../Codebase"
echo "Codebase copied from shared volume."

echo "Deleting deployment..."
kubectl delete -f ingress.yaml -f service.yaml -f master-pod.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml