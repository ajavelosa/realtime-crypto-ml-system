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
