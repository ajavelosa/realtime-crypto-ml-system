port-forward-kafka:
	kubectl port-forward -n kafka svc/kafka-ui 8182:8080

port-forward-risingwave:
	kubectl port-forward -n risingwave svc/risingwave 4567:4567

port-forward-minio:
	kubectl port-forward -n risingwave svc/risingwave-minio 9001:9001

open-psql:
	psql -h localhost -p 4567 -U root -d dev

lint:
	ruff check .

format:
	ruff format .

################################################################################
## Development
################################################################################

# Builds the base Docker image with common dependencies
build-base:
	docker build -t real-time-ml-system-4-base:latest -f docker/Dockerfile.base .

# Runs the trades service as a standalone Python application (not Dockerized)
dev:
	uv run services/${service}/src/${service}/main.py

# Builds the trades service Docker image
build-for-dev: build-base
	docker build -t ${service}:dev -f docker/${service}.Dockerfile .

# Pushes the trades service Docker image to the Docker registry in our kind cluster
push-for-dev:
	kind load docker-image ${service}:dev --name rwml-34fa

# Deploys the trades service to our kind cluster
deploy-for-dev: build-for-dev push-for-dev
	kubectl delete -f deployments/dev/${service}/${service}.yaml --ignore-not-found=true
	kubectl apply -f deployments/dev/${service}/${service}.yaml

################################################################################
## Production
################################################################################

# Builds the trades service Docker image
build-and-push-for-prod: build-base
	@BUILD_DATE=$$(date +%s); \
	echo "BUILD_DATE: $$BUILD_DATE"; \
	docker buildx build --push --platform linux/amd64 -t "ghcr.io/ajavelosa/${service}:0.0.1-beta.$$BUILD_DATE" -f docker/${service}.Dockerfile .

deploy-for-prod:
	kubectl delete -f deployments/prod/${service}/${service}.yaml --ignore-not-found=true
	kubectl apply -f deployments/prod/${service}/${service}.yaml
