FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies and Streamlit
RUN pip install --no-cache-dir -r requirements.txt streamlit==1.33.0

# Copy application code
COPY main.py .
COPY logistic_pipeline_model.pkl .
COPY streamlit_app.py .
COPY entrypoint.sh .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose the port Render expects
EXPOSE 10000

# Use the PORT env variable for Streamlit (default to 10000)
ENV PORT=10000

ENTRYPOINT ["./entrypoint.sh"]

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1 