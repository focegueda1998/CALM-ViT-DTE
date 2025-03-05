#/bin/bash

echo "Copying Codebase from shared volume..."
export BACKUP_DIR="../Codebase-$(uuidgen)"
mkdir -p $BACKUP_DIR
kubectl cp --retries=3 master:config/Codebase/ "$BACKUP_DIR"
echo "Codebase copied from shared volume."

echo "Deleting deployment..."
kubectl delete -f service.yaml -f pvc.yaml -f master-pod.yaml -f slave01-pod.yaml -f slave02-pod.yaml -f slave03-pod.yaml