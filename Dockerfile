FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

# asyncmy ships Cython extensions; on aarch64 there is no pre-built wheel,
# so gcc is required to compile from source during pip install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && pip install --no-cache-dir . \
    && apt-get purge -y gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy remaining project files
COPY webui/ webui/
COPY . .

# Expose API and Streamlit ports
EXPOSE 8080 8501

# Default command: start the API server
CMD ["coscientist", "serve"]
