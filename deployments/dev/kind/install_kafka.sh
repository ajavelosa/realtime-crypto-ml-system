#!/bin/bash

kubectl create namespace kafka
kubectl create -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka
# Wait for the operator pod to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=strimzi-cluster-operator -n kafka --timeout=300s
kubectl apply -f manifests/kafka-e11b.yaml
