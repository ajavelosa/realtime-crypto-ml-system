.PHONY: create-table

create-table:
	psql \
        -h $(PG_HOST) \
		-p $(PG_PORT) \
		-d $(PG_DATABASE) \
		-U $(PG_USER) \
		-v view_name="$(PG_VIEW_NAME)" \
		-v table_name="$(PG_TABLE_NAME)" \
		-f queries/create_latest_predictions_view.sql

build:
	docker build -t prediction-api:dev .

run:
	docker run -p ${API_PORT}:${API_PORT} prediction-api \
		-e PG_HOST=${PG_HOST} \
		-e PG_PORT=${PG_PORT} \
		-e PG_DATABASE=${PG_DATABASE} \
		-e PG_USER=${PG_USER} \
		-e PG_PASSWORD=${PG_PASSWORD} \
		-e PG_VIEW_NAME=${PG_VIEW_NAME} \
		-e RUST_LOG=info \
		prediction-api:dev
