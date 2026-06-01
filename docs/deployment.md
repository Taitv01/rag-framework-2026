# Deployment Guide

## Overview

This guide covers deploying the Ultimate RAG Framework in various environments:
- Local development
- Docker containers
- Cloud platforms (AWS, GCP, Azure)
- Production deployments

## Local Development

### Prerequisites

- Python 3.10+
- pip or poetry

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ultimate-rag.git
cd ultimate-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Running Examples

```bash
# Basic RAG
python examples/01_naive_rag.py

# Advanced RAG
python examples/02_advanced_rag.py

# Agentic RAG
python examples/03_agentic_rag.py

# Production RAG
python examples/04_production_rag.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_rag_pipeline.py
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["python", "examples/04_production_rag.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  rag-app:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEFAULT_LLM_MODEL=gpt-4o-mini
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  qdrant_data:
```

### Build and Run

```bash
# Build image
docker build -t ultimate-rag .

# Run container
docker run -it --env-file .env ultimate-rag

# Run with docker-compose
docker-compose up -d
```

## Cloud Deployment

### AWS

#### Using EC2

```bash
# Connect to EC2
ssh -i key.pem ec2-user@your-instance

# Install dependencies
sudo yum update -y
sudo yum install python3 -y
pip3 install -r requirements.txt

# Run application
python3 examples/04_production_rag.py
```

#### Using Lambda

```python
# lambda_function.py
import json
from src.rag import NaiveRAG

rag = None

def lambda_handler(event, context):
    global rag

    if rag is None:
        rag = NaiveRAG()
        rag.add_documents(["s3://bucket/documents/"])

    question = event.get("question", "")
    answer = rag.query(question)

    return {
        "statusCode": 200,
        "body": json.dumps({"answer": answer})
    }
```

#### Using ECS

```json
{
  "family": "rag-task",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "rag-container",
      "image": "yourusername/ultimate-rag:latest",
      "memory": 4096,
      "cpu": 2,
      "environment": [
        {"name": "OPENAI_API_KEY", "value": "your-key"}
      ]
    }
  ]
}
```

### Google Cloud Platform

#### Using Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ultimate-rag
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: '10'
    spec:
      containers:
      - image: gcr.io/project-id/ultimate-rag
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
```

#### Deploy to Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/project-id/ultimate-rag

# Deploy
gcloud run deploy ultimate-rag \
  --image gcr.io/project-id/ultimate-rag \
  --platform managed \
  --region us-central1 \
  --memory 4Gi
```

### Azure

#### Using Azure Container Instances

```yaml
# azure-container.yaml
apiVersion: '2021-07-01'
location: eastus
name: ultimate-rag
properties:
  containers:
  - name: rag-container
    properties:
      image: yourusername/ultimate-rag:latest
      resources:
        requests:
          memoryInGB: 4
          cpu: 2
      environmentVariables:
      - name: OPENAI_API_KEY
        secureValue: your-key
```

#### Deploy to Azure

```bash
# Create resource group
az group create --name rag-rg --location eastus

# Deploy container
az container create \
  --resource-group rag-rg \
  --file azure-container.yaml
```

## Production Deployment

### Best Practices

1. **Use Production Vector Store**
   ```python
   # Use Qdrant or Weaviate for production
   rag = AdvancedRAG(
       vector_store_provider="qdrant",
       url="http://qdrant:6333"
   )
   ```

2. **Enable Caching**
   ```python
   # Use Redis for distributed caching
   from src.utils.cache import Cache

   cache = Cache(max_size=10000, ttl=3600)
   ```

3. **Set Up Monitoring**
   ```python
   # Enable logging
   from src.utils.logger import setup_logger

   logger = setup_logger("production", level="INFO")
   ```

4. **Use Environment Variables**
   ```bash
   # Never hardcode API keys
   export OPENAI_API_KEY=sk-...
   ```

5. **Implement Health Checks**
   ```python
   def health_check():
       try:
           rag.query("test")
           return {"status": "healthy"}
       except Exception as e:
           return {"status": "unhealthy", "error": str(e)}
   ```

### Scaling Considerations

1. **Horizontal Scaling**
   - Use stateless services
   - Share vector store across instances
   - Use distributed caching

2. **Vertical Scaling**
   - Increase memory for large document sets
   - Use GPU for embeddings
   - Optimize chunk sizes

3. **Cost Optimization**
   - Use appropriate models
   - Implement caching
   - Batch processing

### Security

1. **API Key Management**
   - Use secrets manager (AWS Secrets Manager, Google Secret Manager)
   - Rotate keys regularly
   - Monitor API usage

2. **Data Security**
   - Encrypt data at rest
   - Use HTTPS for API calls
   - Implement access controls

3. **Network Security**
   - Use VPC for cloud deployments
   - Implement firewall rules
   - Monitor network traffic

## Monitoring

### Metrics to Track

1. **Performance**
   - Query latency
   - Retrieval accuracy
   - Cache hit rate

2. **Cost**
   - API calls
   - Token usage
   - Storage costs

3. **Health**
   - Error rates
   - Service availability
   - Resource usage

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler()
    ]
)
```

### Alerting

Set up alerts for:
- High error rates
- Increased latency
- API quota exhaustion
- Service downtime

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API key is set correctly
   - Check API key permissions
   - Ensure API key is not expired

2. **Memory Issues**
   - Reduce chunk size
   - Use streaming for large documents
   - Implement pagination

3. **Performance Issues**
   - Enable caching
   - Use appropriate models
   - Optimize retrieval parameters

4. **Connection Issues**
   - Check network connectivity
   - Verify service endpoints
   - Check firewall rules

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose mode
rag = NaiveRAG(verbose=True)
```
