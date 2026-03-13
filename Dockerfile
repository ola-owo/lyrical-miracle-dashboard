FROM python:3.13-slim
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY . ./
# COPY .python-version pyproject.toml ./
RUN uv sync
# COPY .streamlit/ .streamlit/
# COPY lyrical-miracle-dashboard/ dashboard/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["uv", "run", "streamlit", "run", "dashboard/Home.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
