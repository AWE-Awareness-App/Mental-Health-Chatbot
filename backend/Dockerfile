FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies (including Azure Speech SDK requirements)
# Using Bullseye which has OpenSSL 1.1 natively for Azure Speech SDK compatibility
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    libssl-dev \
    ca-certificates \
    libasound2 \
    libssl1.1 \
    build-essential \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/knowledge_base /app/logs

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
