# base image: python 3.13 with uv
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_NO_CACHE=1
ENV UV_LOCKED=1
ENV UV_NO_DEV=1

WORKDIR /app

# 1st sync install dependencies only
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project
# 2nd sync installs the project itself
COPY . ./
RUN uv sync

ENV STREAMLIT_SERVER_HEADLESS=true
ENTRYPOINT ["sh", "-c", \
    "exec uv run streamlit run pages/Home.py \
    --server.port=$PORT \"${@}\"", "--"]
