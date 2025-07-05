# Authentication and Credentials for GCP Instructions

## Overview

This document provides a comprehensive guide to setting up Google Cloud Platform (GCP) authentication for the Multi-Agent Research Platform, specifically addressing the challenges and solutions when working in a Windows Subsystem for Linux (WSL) environment with Google Agent Development Kit (ADK) and Vertex AI integration.

## Authentication Architecture

Our platform uses **Application Default Credentials (ADC)** as the primary authentication method for Google Cloud services. This approach provides:

- **Security**: No hardcoded API keys or service account files in code
- **Flexibility**: Automatic credential discovery across different environments
- **Best Practices**: Follows Google Cloud recommended authentication patterns
- **Scalability**: Works in development, staging, and production environments

### Authentication Flow

```
User Account → ADC → Google Auth Libraries → Vertex AI API → ADK Integration
```

## Initial Setup Process

### 1. Google Cloud Project Configuration

Our project is configured as follows:
- **Project ID**: `cs-poc-czxf7xbmmrua9yw8mrrkrn0`
- **Project Name**: `adk-deployment-samples`
- **Organization**: ClearSpring Consulting Group

### 2. Required APIs and Services

Ensure the following APIs are enabled in your GCP project:

```bash
# Check enabled services
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
gcloud services list --enabled --filter="name:generativelanguage.googleapis.com"

# Enable if not already enabled
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

**Required Services:**
- **Vertex AI API** (`aiplatform.googleapis.com`) - For ADK integration
- **Generative Language API** (`generativelanguage.googleapis.com`) - For Gemini models

### 3. IAM Permissions and Roles

#### Required Roles for User Account

Your user account (`kevin@clearspringcg.com`) needs the following IAM roles:

1. **Vertex AI User** (`roles/aiplatform.user`)
   - Allows access to Vertex AI services
   - Required for ADK agent creation and execution

2. **ML Developer** (`roles/ml.developer`) 
   - Provides broader ML platform access
   - Includes model deployment and endpoint management

3. **Project Viewer** (`roles/viewer`)
   - Basic project access and resource visibility

#### Verification Commands

```bash
# Check your current IAM roles
gcloud projects get-iam-policy cs-poc-czxf7xbmmrua9yw8mrrkrn0 \
  --flatten="bindings[].members" \
  --filter="bindings.members:kevin@clearspringcg.com" \
  --format="table(bindings.role)"

# Check if you have Vertex AI access specifically
gcloud ai models list --region=us-central1 --project=cs-poc-czxf7xbmmrua9yw8mrrkrn0
```

## WSL-Specific Authentication Challenges and Solutions

### Challenge 1: Browser Authentication Issues

**Problem**: WSL cannot directly open browsers for OAuth flows, leading to authentication failures.

**Error Encountered**:
```
ERROR: (gcloud.auth.application-default.login) 
Invalid parameter value for code_challenge_method: 'S25' is not a valid CodeChallengeMethod
Error 400: invalid_request
```

**Root Cause**: Parameter corruption in the OAuth PKCE flow when redirecting through WSL.

**Solution**: Use manual browser authentication:

```bash
# Use no-browser flag for manual authentication
gcloud auth application-default login --no-browser
```

This provides a URL to copy into your Windows browser manually.

### Challenge 2: ADC Token Refresh Issues

**Problem**: ADC tokens expire and cannot be refreshed automatically in WSL.

**Error Encountered**:
```
ERROR: There was a problem refreshing your current auth tokens: 
Reauthentication failed. cannot prompt during non-interactive execution.
```

**Solution**: Multiple fallback approaches:

#### Option A: Manual Token Refresh
```bash
# Clear existing tokens
gcloud auth application-default revoke

# Re-authenticate with manual browser
gcloud auth application-default login --no-browser
```

#### Option B: Use Existing User Credentials
```bash
# Use your already authenticated user account for ADC
gcloud auth application-default login --account kevin@clearspringcg.com
```

#### Option C: Windows-based Authentication
If WSL continues to fail, authenticate from Windows PowerShell:
```powershell
# Run from Windows (not WSL)
gcloud auth application-default login
```

The credentials are shared between Windows and WSL.

### Challenge 3: Environment Variable Configuration

**Problem**: Multiple `.env` files and shell environment variables causing conflicts.

**Issues Encountered**:
- Shell variables overriding `.env` file settings
- Confusion about which `.env` file is loaded
- Incorrect `GOOGLE_GENAI_USE_VERTEXAI` values

**Solution**: Proper environment configuration:

#### Correct .env File Location
The platform loads `.env` from the project root:
```
/home/kjdrag/lrepos/multi-agent-adk-integration/.env
```

#### Required Environment Variables
```bash
# Essential settings for Vertex AI + ADC
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_PROJECT=cs-poc-czxf7xbmmrua9yw8mrrkrn0

# NOT needed when using ADC (comment out or remove)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# GOOGLE_API_KEY=your_api_key_here
```

#### Environment Validation
```bash
# Check shell environment for conflicts
env | grep -i google

# Verify .env loading in Python
python -c "
from dotenv import load_dotenv
load_dotenv()
import os
print(f'GOOGLE_GENAI_USE_VERTEXAI: {os.getenv(\"GOOGLE_GENAI_USE_VERTEXAI\")}')
print(f'GOOGLE_CLOUD_PROJECT: {os.getenv(\"GOOGLE_CLOUD_PROJECT\")}')
"
```

## Authentication Verification Process

### Step 1: Basic Authentication Check
```bash
# List authenticated accounts
gcloud auth list

# Verify active project
gcloud config get-value project
```

Expected output:
```
     Credentialed Accounts
ACTIVE  ACCOUNT
*       kevin@clearspringcg.com

cs-poc-czxf7xbmmrua9yw8mrrkrn0
```

### Step 2: ADC Token Verification
```bash
# Test ADC token generation (may fail in WSL - this is normal)
gcloud auth application-default print-access-token
```

### Step 3: Google Auth Libraries Test
```bash
# Test Python Google Auth libraries (this is what matters for our platform)
python -c "
import google.auth
try:
    credentials, project = google.auth.default()
    print(f'✅ Google auth working with project: {project}')
    print(f'✅ Credentials type: {type(credentials).__name__}')
except Exception as e:
    print(f'❌ Google auth failed: {e}')
"
```

Expected output:
```
✅ Google auth working with project: cs-poc-czxf7xbmmrua9yw8mrrkrn0
✅ Credentials type: Credentials
```

### Step 4: Platform Integration Test
```bash
# Start the platform to verify ADK integration
python -m src.streamlit.launcher -e development

# Look for successful ADK agent creation in logs:
# "Created ADK agent for Research Specialist"
# "Created ADK runner for Research Specialist"
```

## Periodic Maintenance and Token Refresh

### When to Refresh Credentials

ADC tokens typically expire after **1 hour**. However, Google Auth libraries automatically handle refresh in most cases. Manual refresh is needed when:

1. You see authentication errors in platform logs
2. ADK agents fail to initialize with 401 errors
3. After long periods of inactivity (>7 days)

### Refresh Process

#### Method 1: Automatic Refresh (Preferred)
The platform automatically refreshes tokens when needed. No manual intervention required.

#### Method 2: Manual Refresh (When Method 1 Fails)
```bash
# From WSL (if working)
gcloud auth application-default login --no-browser

# From Windows PowerShell (if WSL fails)
gcloud auth application-default login
```

#### Method 3: User Account Re-authentication
```bash
# Re-authenticate your user account
gcloud auth login --account kevin@clearspringcg.com

# Then refresh ADC
gcloud auth application-default login --account kevin@clearspringcg.com
```

### Monitoring Authentication Health

#### Platform Health Check
```bash
# Check platform authentication status
curl http://localhost:8081/health

# Expected response includes Google Auth status
```

#### Log Monitoring
```bash
# Monitor for authentication errors
grep -i "401\|unauthorized\|authentication" logs/runs/*/error.log

# Monitor ADK agent creation success
grep "Created ADK agent" logs/runs/*/info.log
```

## Troubleshooting Common Issues

### Issue 1: 401 UNAUTHENTICATED Errors

**Symptoms**:
```
googleapiclient.errors.HttpError: <HttpError 401 when requesting ... returned "Request had invalid authentication credentials.">
```

**Diagnosis**:
```bash
# Check if using correct authentication method
python -c "
import os
print(f'Using Vertex AI: {os.getenv(\"GOOGLE_GENAI_USE_VERTEXAI\")}')
print(f'API Key set: {\"SET\" if os.getenv(\"GOOGLE_API_KEY\") else \"NOT SET\"}')
print(f'Credentials file: {os.getenv(\"GOOGLE_APPLICATION_CREDENTIALS\", \"NOT SET\")}')
"
```

**Solutions**:
1. Ensure `GOOGLE_GENAI_USE_VERTEXAI=True`
2. Remove any `GOOGLE_API_KEY` environment variables
3. Refresh ADC tokens

### Issue 2: Project Access Denied

**Symptoms**:
```
ERROR: (gcloud.projects.describe) User [kevin@clearspringcg.com] does not have permission to access project [cs-poc-czxf7xbmmrua9yw8mrrkrn0]
```

**Solution**:
Contact GCP project administrator to verify IAM roles assignment.

### Issue 3: WSL Browser Issues

**Symptoms**:
- Browser fails to open during `gcloud auth` commands
- OAuth redirects fail with parameter errors

**Solutions**:
1. Always use `--no-browser` flag
2. Authenticate from Windows if WSL fails
3. Copy/paste URLs manually into Windows browser

### Issue 4: Environment Variable Conflicts

**Symptoms**:
- Platform uses wrong authentication method
- Shell variables override `.env` file

**Diagnosis**:
```bash
# Check for conflicting shell variables
env | grep -E "GOOGLE_(API_KEY|APPLICATION_CREDENTIALS|GENAI_USE_VERTEXAI)"
```

**Solution**:
```bash
# Clear conflicting shell variables
unset GOOGLE_API_KEY
unset GOOGLE_APPLICATION_CREDENTIALS

# Restart shell or Claude session to ensure clean environment
```

## Best Practices and Recommendations

### 1. Development Workflow
- **Always use ADC** over service account keys or API keys
- **Keep `.env` file clean** with only necessary variables
- **Monitor logs** for authentication issues
- **Test authentication** after environment changes

### 2. Security Considerations
- **Never commit** service account keys to repositories
- **Rotate credentials** periodically if using service accounts
- **Use minimal IAM permissions** (principle of least privilege)
- **Monitor access logs** for unusual activity

### 3. WSL-Specific Best Practices
- **Authenticate from Windows** when WSL browser issues occur
- **Use `--no-browser`** flag for all `gcloud auth` commands
- **Keep credentials synced** between Windows and WSL
- **Test both environments** to ensure consistency

### 4. Platform Integration
- **Verify environment variables** before starting platform
- **Monitor ADK agent creation** for authentication success
- **Use platform health checks** to monitor auth status
- **Keep backup authentication methods** ready

## Environment Variables Reference

### Required Variables
```bash
# Core authentication settings
GOOGLE_GENAI_USE_VERTEXAI=True           # Use Vertex AI instead of direct API
GOOGLE_CLOUD_PROJECT=cs-poc-czxf7xbmmrua9yw8mrrkrn0  # GCP project ID

# Application settings
ENVIRONMENT=development                   # Platform environment
LOG_LEVEL=INFO                           # Logging level
```

### Variables to Avoid/Remove
```bash
# Don't use these with ADC
GOOGLE_API_KEY=...                       # Conflicts with Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=...       # Use ADC instead
```

### Platform-Specific Variables
```bash
# Service backends
SESSION_SERVICE_BACKEND=database         # Session management
MEMORY_SERVICE_BACKEND=database          # Memory storage
ARTIFACT_SERVICE_BACKEND=local           # Artifact storage

# Performance settings
MAX_CONCURRENT_AGENTS=5                  # Agent concurrency
DEFAULT_TIMEOUT_SECONDS=300              # Task timeout
```

## Conclusion

The authentication setup for our Multi-Agent Research Platform successfully uses Google Cloud ADC with Vertex AI integration. While WSL presents some challenges for the `gcloud` CLI tools, the underlying Google Auth libraries work correctly, enabling seamless ADK integration.

Key success factors:
1. **Proper IAM role assignment** for Vertex AI access
2. **Correct environment variable configuration** with Vertex AI enabled
3. **WSL-aware authentication procedures** using manual browser flows
4. **Regular monitoring and maintenance** of credential health

This setup provides secure, scalable authentication that follows Google Cloud best practices while working effectively in our WSL development environment.