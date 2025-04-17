port-forward:
	kubectl -n kafka port-forward svc/kafka-ui 8182:8080

# Runs the trades service as a standalone Python application (not Dockerized)
dev:
	uv run services/${service}/src/${service}/main.py

# Builds the trades service Docker image
build:
	docker build -t ${service}:dev -f docker/${service}.Dockerfile .

# Pushes the trades service Docker image to the Docker registry in our kind cluster
push:
	kind load docker-image ${service}:dev --name rwml-34fa

# Deploys the trades service to our kind cluster
deploy: build push
	kubectl delete -f deployments/dev/${service}/${service}.yaml --ignore-not-found=true
	kubectl apply -f deployments/dev/${service}/${service}.yaml

lint:
	ruff check .

format:
	ruff format .
