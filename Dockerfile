FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Expose API port
EXPOSE 8080

# Default command: start the API server
CMD ["coscientist", "serve"]
