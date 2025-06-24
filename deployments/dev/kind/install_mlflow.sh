#!/bin/bash
kubectl create namespace mlflow
kubectl apply -f manifests/mlflow-minio-secret.yaml
helm upgrade --install --create-namespace --wait mlflow oci://registry-1.docker.io/bitnamicharts/mlflow --namespace=mlflow --values manifests/mlflow-values.yaml
