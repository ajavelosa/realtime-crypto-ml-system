#!/bin/bash

kubectl create namespace kafka
kubectl create -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka
kubectl apply -f manifests/kafka-e11b.yaml

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
kubectl wait --for=condition=ready pod -l strimzi.io/cluster=kafka-e11b -n kafka --timeout=300s
