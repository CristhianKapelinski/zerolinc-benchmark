FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["uv", "run", "zerolinc"]
CMD ["--help"]
