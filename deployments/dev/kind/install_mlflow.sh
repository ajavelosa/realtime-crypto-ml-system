#!/bin/bash
kubectl create namespace mlflow

# Generate MLflow MinIO credentials (most concise approach)
echo "🔐 Generating MLflow MinIO access credentials..."
ACCESS_KEY="mlflow$(openssl rand -hex 6)"  # 6 hex chars = 12 chars + "mlflow" = 18 chars total
SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Create/update the Kubernetes secret
kubectl delete secret mlflow-minio-secret -n mlflow --ignore-not-found=true
kubectl create secret generic mlflow-minio-secret \
  --namespace=mlflow \
  --from-literal=AccessKeyID="$ACCESS_KEY" \
  --from-literal=SecretKey="$SECRET_KEY"

echo "✅ Generated Access Key: $ACCESS_KEY (${#ACCESS_KEY} chars)"

# Register the access keys with MinIO itself
echo "🔑 Registering access keys with MinIO..."

# Wait for MinIO to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=minio -n risingwave --timeout=60s

# Create a temporary script for MinIO setup
cat > /tmp/minio-setup.sh << EOF
#!/bin/sh
mc alias set minio http://risingwave-minio.risingwave.svc.cluster.local:9000 admin minio-D0408AC0
mc admin user svcacct add minio admin --access-key $ACCESS_KEY --secret-key $SECRET_KEY --name mlflow-service-account
echo "MLflow service account created successfully in MinIO!"
echo "Access Key: $ACCESS_KEY"
EOF

# Run the setup script in a temporary pod
kubectl run minio-key-setup --rm -i --restart=Never --image=minio/mc:latest --command -- sh -c "$(cat /tmp/minio-setup.sh)"

# Clean up
rm /tmp/minio-setup.sh

echo "🎉 MLflow MinIO setup complete!"
echo "  - Access Key: $ACCESS_KEY"
echo "  - Keys are now visible in MinIO UI under Access Keys"

# Install MLflow
helm upgrade --install --create-namespace --wait mlflow oci://registry-1.docker.io/bitnamicharts/mlflow --namespace=mlflow --values manifests/mlflow-values.yaml

# Restart MLflow deployment to pick up the new credentials
echo "🔄 Restarting MLflow to use new credentials..."
kubectl rollout restart deployment mlflow-tracking -n mlflow
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mlflow -n mlflow --timeout=60s

echo "✅ MLflow is ready with auto-generated MinIO credentials!"
echo "   You can now run 'uv run train.py' and MLflow will save artifacts to MinIO successfully."
