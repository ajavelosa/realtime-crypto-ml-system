#!/bin/bash

# Builds a docker image for the given dockerfile and pushes it to the docker registry

service=$1
env=$2

# Check if user provided the image name and env
if [ -z "$service" ]; then
    echo "Error: Usage: $0 <service> <env>"
    exit 1
fi

if [ -z "$env" ]; then
    echo "Error: Usage: $0 <service> <env>"
    exit 1
fi

# Check that env is either "dev" or "prod"
if [ "$env" != "dev" ] && [ "$env" != "prod" ]; then
    echo "env must be either dev or prod"
    exit 1
fi

BUILD_DATE=$(date +%s)
echo "BUILD_DATE: $BUILD_DATE"

push_image_to_github() {
    local service=$1
    local env=$2
    local base_image=$3
    local dockerfile="docker/${service}.Dockerfile"

    # Special case for base image
    if [ "$service" == "base" ]; then
        dockerfile="docker/Dockerfile.base"
    fi

    DOCKER_BUILDKIT=1 docker buildx build --push \
        --platform linux/amd64 \
        -t ghcr.io/ajavelosa/${service}:0.1.5-beta.${BUILD_DATE} \
        -t ghcr.io/ajavelosa/${service}:latest \
        ${base_image:+--build-arg BASE_IMAGE=${base_image}} \
        --label org.opencontainers.image.revision=$(git rev-parse HEAD) \
        --label org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
        --label org.opencontainers.image.url="https://github.com/ajavelosa/real-time-crypto-ml-system/${dockerfile}" \
        --label org.opencontainers.image.title="${service}" \
        --label org.opencontainers.image.description="${service} Dockerfile" \
        --label org.opencontainers.image.licenses="" \
        --label org.opencontainers.image.source="https://github.com/ajavelosa/real-time-crypto-ml-system" \
        -f ${dockerfile} .
}

# Function to build base image
build_base_image() {
    local env=$1
    echo "Building base image for ${env}..."

    if [ "$env" == "dev" ]; then
        docker build -t base:latest -f docker/Dockerfile.base .
    else
        push_image_to_github "base" "prod" ""
    fi
}

# Function for dev build (fully local)
build_dev() {
    echo "Building image ${service} for dev..."
    build_base_image "dev"
    docker build -t ${service}:dev \
        --build-arg BASE_IMAGE=base:latest \
        -f docker/${service}.Dockerfile .
    kind load docker-image ${service}:dev --name rwml-34fa
}

# Function for prod build (uses GitHub)
build_prod() {
    echo "Building image ${image_name} for prod..."
    build_base_image "prod"
    push_image_to_github "${image_name}" "prod" "ghcr.io/ajavelosa/base:latest"
}

# Main execution
if [ "$env" == "dev" ]; then
    build_dev
elif [ "$env" == "prod" ]; then
    build_prod
else
    echo "Error: Invalid environment. Use 'dev' or 'prod'"
    exit 1
fi
