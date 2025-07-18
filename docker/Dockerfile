# syntax=docker/dockerfile:1.7-labs
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS app
ENV TZ=America/New_York
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
# Copy each directory separately for better layer caching
COPY app/main.py .
COPY app/assets/ ./assets/
COPY app/services/ ./services/
COPY app/utils/ ./utils/
COPY app/tools/ ./tools/
COPY app/models/ ./models/
COPY app/controllers/ ./controllers/
COPY app/ui/ ./ui/
COPY app/.streamlit/ ./.streamlit/
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import requests; requests.get('http://0.0.0.0:8501/_stcore/health').raise_for_status()"]
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

####
FROM app AS docs
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /docs
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra streamlit-docs
COPY app/docs/ /docs
ENV PATH="/docs/.venv/bin:$PATH"
EXPOSE 8001
RUN apt-get update && apt-get install -y curl --no-install-recommends --quiet
ENTRYPOINT ["mkdocs"]

####
FROM app AS jupyter
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV NODE_MAJOR=18
ENV PATH="/app/.venv/bin:$PATH"
RUN apt-get update && apt-get install -y curl ca-certificates gnupg --no-install-recommends --quiet
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
RUN apt-get update && apt-get install nodejs -y --no-install-recommends --quiet
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra jupyterlab
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension \
    && echo '{ "theme": "Darcula" }' > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings \
    && jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
ENTRYPOINT ["jupyter", "lab", "--ServerApp.ip=0.0.0.0", "--ServerApp.port=8888", "--ServerApp.allow_root=True", "--ServerApp.token=''", "--ServerApp.password=''", "--no-browser"]
