FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:0.4.4 /uv /bin/uv

ARG name=ai-serverless-samples
ARG workdir=/workspaces/${name}
WORKDIR ${workdir}

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc python3-dev libraqm-dev libgl1-mesa-dev rsync git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
# ADD . ./
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen
