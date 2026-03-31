FROM python:3.10.16-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/minimind

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel uv

COPY pyproject.toml ./

RUN uv sync --no-install-project \
    && uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 --index-url ${PYTORCH_INDEX_URL}

COPY . ./

CMD ["bash"]