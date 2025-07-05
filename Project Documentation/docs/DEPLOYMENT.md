# Deployment Guide

This comprehensive deployment guide covers all aspects of deploying the Multi-Agent Research Platform across different environments, from local development to production cloud deployments.

## 🚀 Deployment Overview

The Multi-Agent Research Platform supports multiple deployment strategies:

- **Local Development**: Fast iteration and testing
- **Docker Containers**: Consistent environment packaging
- **Google Cloud Run**: Serverless production deployment
- **Kubernetes**: Container orchestration for scale
- **Traditional VPS**: Virtual private server deployment

## 🛠️ Prerequisites

### System Requirements

**Minimum Requirements**:
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB
- Network: Broadband internet connection

**Recommended Production**:
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- Network: High-speed internet with low latency

### Software Dependencies

**Required**:
- Python 3.9+
- UV package manager (or pip)
- Git
- Docker (for container deployment)

**Optional**:
- Google Cloud SDK (for GCP deployment)
- Kubernetes CLI (for K8s deployment)
- Nginx (for reverse proxy)

## 🏠 Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd multi-agent-research-platform

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
python main.py
```

### Environment Configuration

Create `.env` file:
```env
# Core Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG
PORT=8080

# Google API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=false

# External API Keys
OPENWEATHER_API_KEY=your_openweather_key_here

# Optional: MCP Server Configuration
PERPLEXITY_API_KEY=your_perplexity_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Development Server Options

**FastAPI with ADK** (Recommended):
```bash
# Run with ADK integration
python main.py

# Custom port
PORT=8081 python main.py
```

**Direct uvicorn**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**Streamlit Interface**:
```bash
# Production interface
python src/streamlit/launcher.py

# Development mode
python src/streamlit/launcher.py -e development --reload
```

**Web Debug Interface**:
```bash
# Debug interface
python src/web/launcher.py -e debug

# Production web interface
python src/web/launcher.py -e production
```

## 🐳 Docker Deployment

### Building the Container

**Basic Build**:
```bash
# Build image
docker build -t multi-agent-platform .

# Run container
docker run -p 8080:8080 --env-file .env multi-agent-platform
```

**Multi-stage Build** (Production optimized):
```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

EXPOSE 8080
CMD ["python", "main.py"]
```

```bash
# Build production image
docker build -f Dockerfile.prod -t multi-agent-platform:prod .

# Run with environment file
docker run -p 8080:8080 --env-file .env multi-agent-platform:prod
```

### Docker Compose

**development.yml**:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
      - "8501:8501"  # Streamlit
      - "8081:8081"  # Web interface
    environment:
      - ENVIRONMENT=development
      - GOOGLE_GENAI_USE_VERTEXAI=false
    env_file:
      - .env
    volumes:
      - ./src:/app/src
      - ./sessions.db:/app/sessions.db
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

**production.yml**:
```yaml
version: '3.8'
services:
  app:
    image: multi-agent-platform:prod
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - GOOGLE_GENAI_USE_VERTEXAI=true
    env_file:
      - .env.production
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: always
```

### Docker Commands

```bash
# Development
docker-compose -f development.yml up -d

# Production
docker-compose -f production.yml up -d

# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3

# Update and redeploy
docker-compose pull && docker-compose up -d
```

## ☁️ Google Cloud Run Deployment

### Prerequisites

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Automated Deployment Script

**deploy.sh**:
```bash
#!/bin/bash

# Configuration
PROJECT_ID="your-project-id"
SERVICE_NAME="multi-agent-platform"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Build and push image
echo "Building and pushing image..."
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=true" \
  --set-env-vars="ENVIRONMENT=production" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=1

echo "Deployment complete!"
gcloud run services describe $SERVICE_NAME --region=$REGION
```

### Manual Deployment

**Step 1: Build and Push**:
```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/multi-agent-platform .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/multi-agent-platform
```

**Step 2: Deploy to Cloud Run**:
```bash
gcloud run deploy multi-agent-platform \
  --image gcr.io/YOUR_PROJECT_ID/multi-agent-platform \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true \
  --memory 2Gi \
  --cpu 2
```

### Cloud Run Configuration

**service.yaml**:
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: multi-agent-platform
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/timeout: "300"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/PROJECT_ID/multi-agent-platform
        env:
        - name: GOOGLE_GENAI_USE_VERTEXAI
          value: "true"
        - name: ENVIRONMENT
          value: "production"
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
```

## ⚙️ Kubernetes Deployment

### Namespace and ConfigMap

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: multi-agent-platform
```

**configmap.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: multi-agent-platform
data:
  ENVIRONMENT: "production"
  GOOGLE_GENAI_USE_VERTEXAI: "true"
  LOG_LEVEL: "INFO"
```

### Secrets

**secret.yaml**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: multi-agent-platform
type: Opaque
data:
  GOOGLE_API_KEY: <base64-encoded-key>
  OPENWEATHER_API_KEY: <base64-encoded-key>
```

### Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-platform
  namespace: multi-agent-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multi-agent-platform
  template:
    metadata:
      labels:
        app: multi-agent-platform
    spec:
      containers:
      - name: app
        image: multi-agent-platform:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Service and Ingress

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: multi-agent-platform-service
  namespace: multi-agent-platform
spec:
  selector:
    app: multi-agent-platform
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-agent-platform-ingress
  namespace: multi-agent-platform
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: multi-agent-platform-service
            port:
              number: 80
```

### Kubernetes Deployment Commands

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n multi-agent-platform
kubectl logs -f deployment/multi-agent-platform -n multi-agent-platform

# Scale deployment
kubectl scale deployment multi-agent-platform --replicas=5 -n multi-agent-platform

# Update deployment
kubectl set image deployment/multi-agent-platform app=multi-agent-platform:v2 -n multi-agent-platform
```

## 🖥️ Traditional VPS Deployment

### Server Setup

**Ubuntu/Debian Setup**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip git nginx certbot python3-certbot-nginx

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create application user
sudo useradd -m -s /bin/bash appuser
sudo usermod -aG sudo appuser
```

**Application Deployment**:
```bash
# Switch to app user
sudo su - appuser

# Clone and setup application
git clone <repository-url> /home/appuser/multi-agent-platform
cd /home/appuser/multi-agent-platform

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with production values

# Test application
python main.py
```

### Systemd Service

**/etc/systemd/system/multi-agent-platform.service**:
```ini
[Unit]
Description=Multi-Agent Research Platform
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/home/appuser/multi-agent-platform
Environment=PATH=/home/appuser/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/appuser/.local/bin/python main.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Service Management**:
```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable multi-agent-platform
sudo systemctl start multi-agent-platform

# Check status
sudo systemctl status multi-agent-platform

# View logs
sudo journalctl -u multi-agent-platform -f
```

### Nginx Reverse Proxy

**/etc/nginx/sites-available/multi-agent-platform**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**Enable Site and SSL**:
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/multi-agent-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Setup SSL with Let's Encrypt
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔧 Environment-Specific Configurations

### Development Environment

**.env.development**:
```env
ENVIRONMENT=development
LOG_LEVEL=DEBUG
GOOGLE_GENAI_USE_VERTEXAI=false
ENABLE_DEBUG_INTERFACE=true
ENABLE_HOT_RELOAD=true
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

### Staging Environment

**.env.staging**:
```env
ENVIRONMENT=staging
LOG_LEVEL=INFO
GOOGLE_GENAI_USE_VERTEXAI=true
ENABLE_DEBUG_INTERFACE=true
ENABLE_PERFORMANCE_MONITORING=true
```

### Production Environment

**.env.production**:
```env
ENVIRONMENT=production
LOG_LEVEL=WARNING
GOOGLE_GENAI_USE_VERTEXAI=true
ENABLE_DEBUG_INTERFACE=false
ENABLE_SECURITY_HEADERS=true
ENABLE_RATE_LIMITING=true
MAX_CONCURRENT_REQUESTS=100
```

## 📊 Monitoring and Health Checks

### Health Check Endpoints

```python
# Built-in health checks
GET /health          # Basic health status
GET /health/detailed # Comprehensive health info
GET /status          # System status
```

### Monitoring Setup

**Prometheus Configuration**:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'multi-agent-platform'
  static_configs:
  - targets: ['localhost:8080']
  metrics_path: '/metrics'
```

**Docker Compose with Monitoring**:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔐 Security Considerations

### Production Security Checklist

- [ ] **Environment Variables**: No secrets in code or logs
- [ ] **HTTPS**: SSL/TLS encryption enabled
- [ ] **API Keys**: Secure key management
- [ ] **Rate Limiting**: Prevent abuse
- [ ] **Input Validation**: Sanitize all inputs
- [ ] **CORS**: Proper origin configuration
- [ ] **Headers**: Security headers configured
- [ ] **Firewall**: Restrict network access
- [ ] **Updates**: Regular security updates
- [ ] **Monitoring**: Log suspicious activities

### Security Headers

```python
# Recommended security headers
{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}
```

## 🚨 Disaster Recovery

### Backup Strategies

**Database Backup**:
```bash
# SQLite backup
cp sessions.db sessions_backup_$(date +%Y%m%d_%H%M%S).db

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
cp sessions.db "$BACKUP_DIR/sessions_$DATE.db"
find $BACKUP_DIR -name "sessions_*.db" -mtime +7 -delete
```

**Configuration Backup**:
```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env* *.yml *.yaml
```

### Recovery Procedures

**Application Recovery**:
```bash
# Stop application
sudo systemctl stop multi-agent-platform

# Restore from backup
cp /backups/sessions_latest.db sessions.db

# Restart application
sudo systemctl start multi-agent-platform
```

## 📈 Performance Optimization

### Production Optimizations

**Application Level**:
```python
# Gunicorn configuration
bind = "0.0.0.0:8080"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
keepalive = 60
max_requests = 1000
max_requests_jitter = 100
```

**Database Optimization**:
```sql
-- SQLite optimizations
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = memory;
```

**Caching Strategy**:
```python
# Redis caching configuration
REDIS_URL = "redis://localhost:6379"
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000
```

## 🔄 CI/CD Pipeline

### GitHub Actions

**.github/workflows/deploy.yml**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install uv
        uv sync
    - name: Run tests
      run: python -m pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Cloud Run
      uses: google-github-actions/deploy-cloudrun@v0
      with:
        service: multi-agent-platform
        image: gcr.io/${{ secrets.GCP_PROJECT_ID }}/multi-agent-platform
        region: us-central1
```

---

This comprehensive deployment guide covers all major deployment scenarios and provides the foundation for reliable, scalable production deployments of the Multi-Agent Research Platform.