port-forward-kafka:
	kubectl port-forward -n kafka svc/kafka-ui 8182:8080

port-forward-risingwave:
	kubectl port-forward -n risingwave svc/risingwave 4567:4567

port-forward-minio:
	kubectl port-forward -n risingwave svc/risingwave-minio 9001:9001

port-forward-grafana:
	kubectl port-forward -n monitoring svc/grafana 3000:80

open-psql:
	psql -h localhost -p 4567 -U root -d dev

lint:
	ruff check .

format:
	ruff format .

################################################################################
## Development
################################################################################

# Runs the trades service as a standalone Python application (not Dockerized)
dev:
	uv run services/${service}/src/${service}/main.py

build-and-push:
	./scripts/build-and-push-image.sh ${service} ${env}

deploy:
	./scripts/deploy.sh ${service} ${env}

build-and-deploy: build-and-push deploy
