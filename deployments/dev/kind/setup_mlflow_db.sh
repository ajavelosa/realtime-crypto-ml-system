#!/bin/bash

echo "Setting up MLFlow database and user..."

# Create namespace if it doesn't exist
kubectl create namespace mlflow --dry-run=client -o yaml | kubectl apply -f -

# Run the database setup job
kubectl apply -f manifests/mlflow-db-init.yaml

# Wait for the job to complete
echo "Waiting for database setup to complete..."
kubectl wait --for=condition=complete job/mlflow-db-init -n mlflow --timeout=60s

# Check if the job succeeded
if [ $? -eq 0 ]; then
    echo "✅ Database setup completed successfully!"
    echo "You can now install MLFlow with: ./install_mlflow.sh"
else
    echo "❌ Database setup failed. Check the logs with:"
    echo "kubectl logs job/mlflow-db-init -n mlflow"
    exit 1
fi
