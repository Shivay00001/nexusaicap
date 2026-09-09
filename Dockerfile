FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -e .[all]

# Expose ports for Web UI and MCP
EXPOSE 8080
EXPOSE 8765

# Default command
CMD ["nexusai", "server", "--port", "8080"]
