# Use our optimized base image with all dependencies and code pre-installed
ARG BASE_IMAGE=base:latest
FROM ${BASE_IMAGE}

# Service code is already included in base image, just set working directory
WORKDIR /app

# Run the trades service using the pre-installed virtual environment
CMD ["python", "/app/services/trades/src/trades/main.py"]
