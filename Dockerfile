# base image: python 3.13 with uv
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1st sync install dependencies only
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
# 2nd sync installs the project itself
COPY . ./
RUN uv sync --frozen

ENV STREAMLIT_SERVER_HEADLESS=true
ENTRYPOINT ["sh", "-c", \
    "exec uv run streamlit run lyrical-miracle-dashboard/Home.py \
    --server.port=$PORT \"${@}\"", "--"]
