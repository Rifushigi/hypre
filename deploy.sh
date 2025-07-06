#!/bin/bash

# Hypertension Prediction API Deployment Script

echo "🚀 Deploying Hypertension Prediction API..."

# Check if model file exists
if [ ! -f "logistic_pipeline_model.pkl" ]; then
    echo "❌ Error: logistic_pipeline_model.pkl not found!"
    echo "Please ensure the model file is in the current directory."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# Function to check if port is available
check_port() {
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port 8000 is already in use. Please stop the existing service or use a different port."
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
}

# Function to run with Docker
run_docker() {
    echo "🐳 Running with Docker..."
    docker-compose up --build -d
    echo "✅ API is running on http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
}

# Function to run locally
run_local() {
    echo "💻 Running locally..."
    python main.py
}

# Main deployment logic
echo "🔍 Checking prerequisites..."

# Check port availability
check_port

# Install dependencies
install_dependencies

# Ask user for deployment method
echo ""
echo "Choose deployment method:"
echo "1) Run locally with Python"
echo "2) Run with Docker"
echo "3) Run with Docker Compose"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Starting local deployment..."
        run_local
        ;;
    2)
        echo "Starting Docker deployment..."
        docker build -t hypertension-api .
        docker run -d -p 8000:8000 --name hypertension-api-container hypertension-api
        echo "✅ API is running on http://localhost:8000"
        echo "📚 API Documentation: http://localhost:8000/docs"
        ;;
    3)
        echo "Starting Docker Compose deployment..."
        run_docker
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment completed!"
echo "📋 Quick test:"
echo "   curl http://localhost:8000/health"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔧 To stop the service:"
echo "   - Local: Ctrl+C"
echo "   - Docker: docker stop hypertension-api-container"
echo "   - Docker Compose: docker-compose down" 