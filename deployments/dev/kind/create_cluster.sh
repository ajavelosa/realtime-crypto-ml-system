#!/bin/bash
export KUBECONFIG=~/.kube/config-rwml-dev

# Function to check if a resource exists
check_resource() {
    local resource_type=$1
    local resource_name=$2
    if [ "$resource_type" = "cluster" ]; then
        kind get clusters | grep -q "$resource_name"
    elif [ "$resource_type" = "network" ]; then
        docker network ls | grep -q "$resource_name"
    fi
}

# 1. Delete the cluster if it exists
echo "Checking and deleting the cluster..."
if check_resource "cluster" "rwml-34fa"; then
    kind delete cluster --name rwml-34fa
fi

# 2. Delete the docker network if it exists
echo "Checking and deleting the docker network..."
if check_resource "network" "rwml-34fa-network"; then
    docker network rm rwml-34fa-network
fi

# 3. Create the docker network
echo "Creating the docker network..."
docker network create --subnet 172.100.0.0/16 rwml-34fa-network

# 4. Create the cluster
echo "Creating the cluster..."
KIND_EXPERIMENTAL_DOCKER_NETWORK=rwml-34fa-network kind create cluster --config ./kind-with-portmapping.yaml

# 5. Update certificates in the control plane
echo "Updating certificates..."
docker exec rwml-34fa-control-plane update-ca-certificates

echo "Configuring kubectl..."
# 6. Export the kubeconfig to ensure we have the correct port
kind export kubeconfig --name rwml-34fa
kubectl config use-context kind-rwml-34fa

echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# 7. Install Kafka
echo "Installing Kafka..."
chmod +x ./install_kafka.sh
./install_kafka.sh

# 8. Install Kafka UI
echo "Installing Kafka UI..."
chmod +x ./install_kafka_ui.sh
./install_kafka_ui.sh

# 9. Install RisingWave
echo "Installing RisingWave..."
chmod 755 ./install_risingwave.sh
./install_risingwave.sh

# Wait for both Kafka and Kafka UI in parallel
echo "Waiting for Kafka and Kafka UI to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=kafka -n kafka --timeout=300s &
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=kafka-ui -n kafka --timeout=300s &
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=risingwave -n risingwave --timeout=300s &
wait

echo "Cluster setup complete!"
echo "Kafka UI is available at http://localhost:8182"
echo "Kafka broker is available at localhost:31234"
echo ""
echo "To start port forwarding for Kafka UI, run:"
echo "kubectl port-forward -n kafka svc/kafka-ui 8182:8080"
