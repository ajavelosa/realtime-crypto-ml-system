#!/bin/bash
kubectl delete namespace strimzi
kubectl create namespace strimzi
kubectl create -f 'https://strimzi.io/install/latest?namespace=strimzi' -n strimzi
kubectl apply -f manifests/kafka-c6c8.yaml
