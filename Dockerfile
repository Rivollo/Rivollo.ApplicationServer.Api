FROM python:3.11-slim

ENV UV_SYSTEM_PYTHON=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        gcc \
        nodejs \
        npm \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Ensure uv is on PATH (installed to /root/.local/bin by default)
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./

# Install locked dependencies into system Python using uv
# 1) Export a pinned requirements file from uv.lock
# 2) Install into the system site-packages
RUN uv export --frozen > requirements.txt \
    && uv pip install --system -r requirements.txt \
    && rm -f requirements.txt \
    && apt-get purge -y --auto-remove curl build-essential gcc \
    && rm -rf ~/.cache/uv /var/lib/apt/lists/*

COPY . .

# Install the glTF-Transform toolchain used for Draco GLB compression
# (app/services/glb_compression_service.py shells out to this script).
# `npm install --prefix` does not reliably resolve package.json to the
# target directory, so cd into it instead.
RUN cd scripts/glb_compress && npm install --omit=dev

# Verify USD command-line tools are available after installation
RUN python -c "import pxr; print('USD Python bindings available')" || echo "USD Python bindings not available"
RUN which usdzip || echo "usdzip command not found - will use manual fallback"

ENV PORT=8080
# Document the default container port
EXPOSE 8080

# Use shell form so $PORT expands at runtime
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
