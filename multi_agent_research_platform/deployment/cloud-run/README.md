# Cloud Run Deployment Guide

This directory contains all the necessary files and scripts for deploying the Multi-Agent Research Platform to Google Cloud Run.

## 📁 Directory Structure

```
deployment/cloud-run/
├── README.md              # This file
├── Dockerfile             # Production-optimized Docker image
├── service.yaml           # Cloud Run service configuration
├── deploy.sh              # Automated deployment script
├── setup.sh               # Initial Cloud setup script
└── config/
    ├── staging.env         # Staging environment configuration
    └── production.env      # Production environment configuration
```

## 🚀 Quick Start

### 1. Prerequisites

**Required Tools:**
- Google Cloud SDK (`gcloud`)
- Docker
- Git

**Google Cloud Requirements:**
- Google Cloud Project with billing enabled
- Appropriate IAM permissions
- API access to:
  - Cloud Run
  - Cloud Build  
  - Container Registry
  - Secret Manager

### 2. Initial Setup

Run the setup script to configure your Google Cloud project:

```bash
# For existing project
./setup.sh --project-id your-project-id

# For new project
./setup.sh --project-id your-new-project --create-project --billing-account YOUR_BILLING_ID
```

### 3. Configure API Keys

Update the secrets with your actual API keys:

```bash
# Google AI API Key
echo "your_google_api_key" | gcloud secrets versions add google-api-key --data-file=-

# OpenWeather API Key
echo "your_openweather_key" | gcloud secrets versions add openweather-api-key --data-file=-

# Optional: Additional API keys
echo "your_perplexity_key" | gcloud secrets versions add perplexity-api-key --data-file=-
echo "your_tavily_key" | gcloud secrets versions add tavily-api-key --data-file=-
```

### 4. Deploy to Cloud Run

**Deploy to Staging:**
```bash
./deploy.sh --project-id your-project-id --environment staging
```

**Deploy to Production:**
```bash
./deploy.sh --project-id your-project-id --environment production
```

## 📖 Detailed Usage

### Setup Script (`setup.sh`)

The setup script configures your Google Cloud project for Cloud Run deployment.

**Usage:**
```bash
./setup.sh [OPTIONS]
```

**Key Options:**
- `-p, --project-id`: Google Cloud Project ID (required)
- `-r, --region`: Deployment region (default: us-central1)
- `--create-project`: Create new project if it doesn't exist
- `--billing-account`: Billing account ID for new projects
- `--dry-run`: Show what would be done without executing

**What it does:**
1. Creates or validates Google Cloud project
2. Enables required APIs
3. Creates service account with appropriate permissions
4. Sets up Secret Manager secrets
5. Configures Cloud Build permissions
6. Creates VPC connector for private networking
7. Generates configuration files

**Example:**
```bash
# Setup existing project
./setup.sh --project-id my-existing-project

# Create new project with billing
./setup.sh --project-id my-new-project \
           --create-project \
           --billing-account 123456-789ABC-DEF012 \
           --region europe-west1
```

### Deployment Script (`deploy.sh`)

The deployment script handles building and deploying your application to Cloud Run.

**Usage:**
```bash
./deploy.sh [OPTIONS]
```

**Key Options:**
- `-p, --project-id`: Google Cloud Project ID
- `-e, --environment`: Environment (staging|production)
- `-r, --region`: Deployment region
- `-f, --force`: Skip confirmation prompts
- `-d, --dry-run`: Show what would be done
- `-v, --verbose`: Enable detailed output

**Deployment Process:**
1. Validates prerequisites and configuration
2. Builds optimized Docker image using Cloud Build
3. Deploys to Cloud Run with environment-specific settings
4. Performs health checks
5. Displays deployment summary and service URL

**Examples:**
```bash
# Deploy to staging with confirmation
./deploy.sh --project-id my-project --environment staging

# Force deploy to production without prompts
./deploy.sh -p my-project -e production -f

# Dry run to see what would happen
./deploy.sh -p my-project -e production --dry-run
```

### Service Configuration (`service.yaml`)

Defines the Cloud Run service specification with production and staging configurations.

**Key Features:**
- Automatic scaling (1-50 instances for production, 1-10 for staging)
- Resource allocation (4Gi/4CPU for production, 2Gi/2CPU for staging)
- Health checks and startup probes
- Environment variables and security settings
- VPC networking configuration

**Customization:**
Edit the file to adjust:
- Resource limits and requests
- Scaling parameters
- Environment variables
- Network settings

### Docker Configuration (`Dockerfile`)

Multi-stage Dockerfile optimized for Cloud Run deployment.

**Optimization Features:**
- Multi-stage build for minimal image size
- UV package manager for faster dependency installation
- Non-root user for security
- Health check endpoint
- Production environment configuration

**Build Process:**
1. **Builder stage**: Installs dependencies and builds application
2. **Runtime stage**: Creates minimal production image with only necessary components

### Environment Configurations

#### Staging (`config/staging.env`)
- Debug logging enabled
- Reduced resource limits
- Debug interfaces enabled
- Extended timeouts for development

#### Production (`config/production.env`)
- Warning-level logging
- Optimized resource allocation
- Security features enabled
- Performance monitoring

## 🔄 CI/CD Integration

### GitHub Actions

The repository includes a comprehensive GitHub Actions workflow (`.github/workflows/deploy-cloud-run.yml`) that provides:

**Automated Testing:**
- Unit and integration tests
- Code quality checks (linting, type checking)
- Security scanning

**Deployment Pipeline:**
- Automatic staging deployment on `develop` branch
- Production deployment on `main` branch
- Manual deployment trigger with environment selection

**Quality Assurance:**
- Performance monitoring
- Health checks after deployment
- Rollback capability on failure

**Setup GitHub Actions:**

1. **Repository Secrets:**
   ```
   GCP_PROJECT_ID: your-project-id
   GCP_SA_KEY: <service-account-json-key>
   GOOGLE_API_KEY: <your-google-api-key>
   OPENWEATHER_API_KEY: <your-openweather-key>
   ```

2. **Service Account Key:**
   ```bash
   # Create and download service account key
   gcloud iam service-accounts keys create sa-key.json \
     --iam-account=multi-agent-platform@PROJECT_ID.iam.gserviceaccount.com
   
   # Add to GitHub secrets as GCP_SA_KEY
   cat sa-key.json | base64 -w 0
   ```

### Cloud Build

The `cloudbuild.yaml` file in the project root provides:
- Automated testing on every build
- Multi-environment deployment
- Integration testing
- Performance monitoring
- Cleanup on failure

## 🔧 Configuration Management

### Environment Variables

The platform supports various configuration methods:

**1. Environment Files:**
- Load from `config/staging.env` or `config/production.env`
- Override with Cloud Run environment variables

**2. Secret Manager:**
- API keys stored securely
- Automatic injection into application

**3. Cloud Run Configuration:**
- Resource limits and scaling
- Network and security settings

### Key Configuration Options

**Agent Settings:**
```env
MAX_CONCURRENT_AGENTS=10
MAX_CONCURRENT_TASKS=20
AGENT_TIMEOUT_SECONDS=120
```

**Performance Settings:**
```env
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_RESPONSE_CACHING=true
CACHE_TTL_SECONDS=1800
```

**Security Settings:**
```env
ENABLE_SECURITY_HEADERS=true
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS_PER_MINUTE=500
```

## 📊 Monitoring and Observability

### Built-in Monitoring

The deployment includes comprehensive monitoring:

**Health Endpoints:**
- `/health`: Basic health status
- `/health/ready`: Readiness probe
- `/health/live`: Liveness probe
- `/status`: Detailed system status

**Metrics Collection:**
- Prometheus metrics at `/metrics`
- Cloud Monitoring integration
- Performance tracking
- Error rate monitoring

**Logging:**
- Structured JSON logging
- Cloud Logging integration
- Error tracking and alerting

### Performance Monitoring

**Load Testing:**
- Automated performance tests in CI/CD
- Artillery-based load testing
- Response time monitoring

**Resource Monitoring:**
- CPU and memory usage
- Request concurrency
- Cold start metrics

## 🛠️ Troubleshooting

### Common Issues

**1. Authentication Errors:**
```bash
# Re-authenticate with gcloud
gcloud auth login
gcloud auth application-default login
```

**2. Permission Denied:**
```bash
# Check IAM permissions
gcloud projects get-iam-policy PROJECT_ID
gcloud iam service-accounts get-iam-policy SERVICE_ACCOUNT_EMAIL
```

**3. Build Failures:**
```bash
# Check Cloud Build logs
gcloud builds list --limit=5
gcloud builds log BUILD_ID
```

**4. Deployment Issues:**
```bash
# Check Cloud Run logs
gcloud run services logs read SERVICE_NAME --region=REGION
```

### Debug Commands

**Service Status:**
```bash
# Check service status
gcloud run services describe SERVICE_NAME --region=REGION

# Get service URL
gcloud run services describe SERVICE_NAME --region=REGION --format="value(status.url)"

# Check revisions
gcloud run revisions list --service=SERVICE_NAME --region=REGION
```

**Health Checks:**
```bash
# Test health endpoint
curl -f https://your-service-url/health

# Test with detailed output
curl -v https://your-service-url/status
```

## 🔒 Security Considerations

### Best Practices

**1. Service Account Security:**
- Use dedicated service account with minimal permissions
- Regular key rotation
- Avoid downloading service account keys when possible

**2. Secret Management:**
- Store API keys in Secret Manager
- Use IAM for secret access control
- Regular secret rotation

**3. Network Security:**
- Use VPC connectors for private networking
- Configure appropriate ingress settings
- Enable Cloud Armor for DDoS protection

**4. Application Security:**
- Enable security headers
- Implement rate limiting
- Use HTTPS only
- Regular security scanning

### Security Checklist

- [ ] Service account follows principle of least privilege
- [ ] API keys stored in Secret Manager
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Regular security updates
- [ ] Monitoring and alerting configured

## 📈 Scaling and Performance

### Scaling Configuration

**Automatic Scaling:**
- Min instances: 1 (staging), 2 (production)
- Max instances: 10 (staging), 50 (production)
- Target concurrency: 5 (staging), 10 (production)

**Resource Allocation:**
- Staging: 2 vCPU, 2Gi memory
- Production: 4 vCPU, 4Gi memory
- Request timeout: 300s (staging), 600s (production)

### Performance Optimization

**1. Cold Start Optimization:**
- Minimum instances configuration
- CPU boost during startup
- Optimized container image

**2. Response Caching:**
- Application-level caching
- CDN integration for static assets
- Database query optimization

**3. Resource Efficiency:**
- Connection pooling
- Lazy loading
- Response compression

## 💰 Cost Optimization

### Cost Management

**1. Resource Optimization:**
- Right-sizing instances
- Appropriate scaling parameters
- Efficient resource utilization

**2. Traffic Management:**
- Request routing optimization
- Load balancing
- Traffic splitting for gradual rollouts

**3. Monitoring Costs:**
- Cloud Billing alerts
- Resource usage tracking
- Cost allocation by environment

### Cost Estimation

**Staging Environment:**
- ~$10-30/month for low traffic
- Scales with usage

**Production Environment:**
- $50-200/month depending on traffic
- Pay-per-use model

## 🤝 Contributing

When contributing to the Cloud Run deployment configuration:

1. Test changes in staging environment first
2. Update documentation for any new features
3. Follow security best practices
4. Update CI/CD workflows as needed
5. Validate changes with dry-run mode

## 📚 Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Container Registry Documentation](https://cloud.google.com/container-registry/docs)

---

For questions or issues with Cloud Run deployment, please refer to the main project documentation or create an issue in the repository.