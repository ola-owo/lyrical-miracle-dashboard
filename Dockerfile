FROM python:3.13-slim
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY . ./
RUN uv sync

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health
ENTRYPOINT uv run streamlit run lyrical-miracle-dashboard/Home.py --server.headless=true --server.port=$PORT
