#!/bin/bash
# Steps:

# 1. Delete the cluster (if it exists, otherwise it will fail)
echo "Deleting the cluster..."
kind delete cluster --name rwml-34fa

# 2. Delete the docker network (if it exists, otherwise it will fail)
echo "Deleting the docker network..."
docker network rm rwml-34fa-network

# 3. Create the docker network
echo "Creating the docker network..."
docker network create --subnet 172.100.0.0/16 rwml-34fa-network

# 4. Create the cluster
echo "Creating the cluster..."
KIND_EXPERIMENTAL_DOCKER_NETWORK=rwml-34fa-network kind create cluster --config ./kind-with-portmapping.yaml

echo "Configuring kubectl..."
export KUBECONFIG=~/.kube/config
# Export the kubeconfig to ensure we have the correct port
kind export kubeconfig --name rwml-34fa
kubectl config use-context kind-rwml-34fa

echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

echo "Installing Kafka..."
chmod +x ./install_kafka.sh
./install_kafka.sh

echo "Waiting for Kafka to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=kafka -n kafka --timeout=300s

echo "Installing Kafka UI..."
chmod +x ./install_kafka_ui.sh
./install_kafka_ui.sh

echo "Waiting for Kafka UI to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=kafka-ui -n kafka --timeout=300s

echo "Setting up port forwarding for Kafka UI..."
kubectl port-forward -n kafka svc/kafka-ui 8182:8080

echo "Cluster setup complete!"
echo "Kafka UI is available at http://localhost:8182"
echo "Kafka broker is available at localhost:31234"
