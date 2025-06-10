# Use our base image with TA-Lib pre-installed
ARG BASE_IMAGE=base:latest
FROM ${BASE_IMAGE}

# Install the project into `/app`
WORKDIR /app

COPY services /app/services

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Run the technical indicators service
CMD ["uv", "run", "/app/services/technical-indicators/src/technical_indicators/main.py"]
