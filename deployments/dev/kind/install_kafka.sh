#!/bin/bash

kubectl create namespace kafka
kubectl create -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka

# Wait for the operator to be ready
echo "Waiting for Strimzi operator to be ready..."
kubectl wait --for=condition=Ready pod -l name=strimzi-cluster-operator -n kafka --timeout=300s

# Wait for service account to be created
echo "Waiting for service account to be created..."
kubectl wait --for=condition=Ready serviceaccount strimzi-cluster-operator -n kafka --timeout=300s

# Apply Kafka cluster configuration
echo "Creating Kafka cluster..."
kubectl apply -f manifests/kafka-e11b.yaml

# Wait for Kafka cluster to be ready
echo "Waiting for Kafka cluster to be ready..."
kubectl wait --for=condition=Ready kafka kafka-e11b -n kafka --timeout=300s

echo "Kafka installation completed!"
