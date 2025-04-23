#!/bin/bash
# Steps:

# 1. Load the environment variables
echo "Loading the environment variables..."
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
direnv allow

# 2. Authenticate with Digital Ocean
echo "Authenticating with Digital Ocean..."
doctl auth init ${DIGITALOCEAN_TOKEN}

# 3. Delete the cluster (if it exists, otherwise it will fail)
echo "Deleting the cluster ${CLUSTER_NAME}."
doctl kubernetes cluster delete ${CLUSTER_NAME}

# 4. Create the cluster
echo "Creating the cluster ${CLUSTER_NAME}."
doctl kubernetes cluster create ${CLUSTER_NAME} \
  --region sgp1 \
  --size s-2vcpu-4gb-amd  \
  --count 2

# 5. Export the kubeconfig to ensure we have the correct port
echo "Configuring kubectl..."
doctl kubernetes cluster kubeconfig save ${CLUSTER_NAME}
kubectl config use-context ${CLUSTER_NAME}

echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# 6. Create image pull secret for GitHub Container Registry
echo "Creating namespace apps..."
kubectl create namespace apps

echo "Creating image pull secret..."
kubectl create secret docker-registry ghcr-secret \
  --namespace=apps \
  --docker-server=ghcr.io \
  --docker-username=${GITHUB_USER} \
  --docker-password=${GITHUB_PAT}

echo "Cluster setup complete!"

# 8. Install Kafka
echo "Installing Kafka..."
chmod +x ./install_kafka.sh
./install_kafka.sh

echo "Waiting for Kafka to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=strimzi -n strimzi --timeout=300s

# 9. Install Kafka UI
doctl compute load-balancer delete $(doctl compute load-balancer list --format ID,Name | grep kafka-ui | awk '{print $1}')
echo "Installing Kafka UI..."
chmod +x ./install_kafka_ui.sh
./install_kafka_ui.sh

echo "Waiting for Kafka UI to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=kafka-ui -n strimzi --timeout=300s

echo "Kafka UI is being exposed through a load balancer..."
echo "Please wait a few minutes for the load balancer to be provisioned and DNS to propagate."
echo "You can check the status with: doctl compute load-balancer list"
