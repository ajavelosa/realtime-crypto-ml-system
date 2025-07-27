# real-time-ml-system-cohort-4

#### Table of contents

1. Clone the repository
2. [OPTIONAL] Build the repository in a devcontainer
2. Run the dev `kind` cluster
    a. `cd deployments/dev/kind`
    b. `chmod 755 create_cluster.sh`
    c. `. ./create_cluster.sh`
3. Forward the ports from `k9s`
4. Set the credentials
    a. `minio`: set the `minio-key` in `localhost:9000`
    b. 
4. Run the services via `make build-and-deploy service={service} env={prod or dev}``
    a. `trades`
    b. `candles`
    c. `technical-indicators`
    d. `training-pipeline`
    e. `prediction-generator`
    f. `prediction-api`
5. Port forward the services
    a. Kafka UI - 8182:8182
    b. Minio - 9000:9000
    c. Grafana - 3000:3000
    d. Risingwave - 4567:4567
    e. Postgres - 5432:5432
    f. Prediction API - 8080:8080
