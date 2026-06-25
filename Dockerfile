FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_LOCKED=1 \
    UV_NO_DEV=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project
COPY . ./
RUN uv sync

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501
EXPOSE $STREAMLIT_SERVER_PORT
ENTRYPOINT ["uv", "run", "--no-sync", "streamlit", "run", "Home.py"]
