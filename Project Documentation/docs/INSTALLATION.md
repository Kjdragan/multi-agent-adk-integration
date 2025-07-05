# Installation Guide

This guide provides comprehensive instructions for installing and setting up the Multi-Agent Research Platform in various environments.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.11+ recommended)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for API calls and package downloads

### Required Accounts and API Keys

Before installation, ensure you have:

1. **Google Cloud Account** (for ADK and Gemini API)
2. **OpenWeather API Key** (for weather functionality)
3. **Optional External Services**:
   - Perplexity API key
   - Tavily API key
   - Brave Search API key

## 🚀 Installation Methods

### Method 1: UV Package Manager (Recommended)

UV is a fast Python package manager that provides better dependency resolution and faster installs.

#### Install UV

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv
```

#### Install the Platform

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-research-platform

# Install dependencies with UV
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows
```

### Method 2: Traditional pip Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-research-platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Docker Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-research-platform

# Build Docker image
docker build -t multi-agent-platform .

# Run with environment file
docker run -p 8080:8080 --env-file .env multi-agent-platform
```

## 🔧 Configuration Setup

### 1. Environment Variables

Create your environment configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the configuration
nano .env  # or your preferred editor
```

### 2. Required Environment Variables

Edit your `.env` file with the following required variables:

```bash
# === Core Configuration ===
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_LOCATION=us-central1

# === Environment Setting ===
# For local development (use direct API)
GOOGLE_GENAI_USE_VERTEXAI=false

# For cloud deployment (use Vertex AI)
# GOOGLE_GENAI_USE_VERTEXAI=true

# === External Services ===
OPENWEATHER_API_KEY=your_openweather_api_key

# === Optional MCP Services ===
PERPLEXITY_API_KEY=your_perplexity_key
TAVILY_API_KEY=your_tavily_key
BRAVE_API_KEY=your_brave_search_key

# === Application Settings ===
PORT=8080
LOG_LEVEL=INFO
```

### 3. Agent-Specific Configuration

Create the agent environment file:

```bash
# Copy agent configuration
cp multitool_agent/.env.example multitool_agent/.env

# Edit agent-specific settings
nano multitool_agent/.env
```

Agent configuration (multitool_agent/.env):

```bash
# === Agent Configuration ===
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=false
OPENWEATHER_API_KEY=your_openweather_api_key

# === Agent Behavior ===
AGENT_MAX_RETRIES=3
AGENT_TIMEOUT_SECONDS=30
AGENT_LOG_LEVEL=INFO
```

## 🔑 API Key Setup

### Google Cloud / Gemini API

#### For Local Development

1. **Get API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

2. **Configure for Local Use**:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   GOOGLE_GENAI_USE_VERTEXAI=false
   ```

#### For Cloud Deployment

1. **Set up Google Cloud Project**:
   ```bash
   # Install Google Cloud CLI
   curl https://sdk.cloud.google.com | bash
   
   # Initialize and authenticate
   gcloud init
   gcloud auth application-default login
   ```

2. **Enable Required APIs**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable run.googleapis.com
   ```

3. **Configure for Cloud Use**:
   ```bash
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_GENAI_USE_VERTEXAI=true
   ```

### OpenWeather API

1. **Sign up** at [OpenWeatherMap](https://openweathermap.org/api)
2. **Get free API key** from your account dashboard
3. **Add to environment**:
   ```bash
   OPENWEATHER_API_KEY=your_openweather_key
   ```

### Optional External Services

#### Perplexity API
1. Sign up at [Perplexity API](https://perplexity.ai)
2. Get API key from dashboard
3. Add to `.env`: `PERPLEXITY_API_KEY=your_key`

#### Tavily API
1. Sign up at [Tavily](https://tavily.com)
2. Get API key from dashboard
3. Add to `.env`: `TAVILY_API_KEY=your_key`

## ✅ Verification

### 1. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.9+

# Check package installation
python -c "import google.generativeai; print('Google AI package installed')"
python -c "import streamlit; print('Streamlit installed')"
python -c "import fastapi; print('FastAPI installed')"
```

### 2. Test Configuration

```bash
# Test environment loading
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Google API Key:', 'Set' if os.getenv('GOOGLE_API_KEY') else 'Missing')
print('OpenWeather Key:', 'Set' if os.getenv('OPENWEATHER_API_KEY') else 'Missing')
"
```

### 3. Quick Functionality Test

Test the basic agent functionality:

```bash
# Test basic agent creation
python -c "
import sys
sys.path.append('src')
from agents import AgentFactory
factory = AgentFactory()
print('Agent factory created successfully')
"
```

## 🚀 First Run

### Start the Streamlit Interface

```bash
# Production interface (recommended for new users)
python src/streamlit/launcher.py

# Development mode with auto-reload
python src/streamlit/launcher.py -e development --reload
```

Access at: http://localhost:8501

### Start the Web Debug Interface

```bash
# Debug interface with monitoring
python src/web/launcher.py -e debug

# Production web interface
python src/web/launcher.py -e production
```

Access at: http://localhost:8081

### Verify Everything is Working

1. **Check the interface loads** without errors
2. **Create a test agent** through the UI
3. **Execute a simple task** (e.g., "What is AI?")
4. **Verify the response** is generated successfully

## 🐛 Common Installation Issues

### Issue: Import Errors

**Problem**: `ModuleNotFoundError` when importing packages

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Google API Authentication

**Problem**: Authentication errors with Google APIs

**Solutions**:
```bash
# For local development - check API key
echo $GOOGLE_API_KEY

# For cloud deployment - check authentication
gcloud auth application-default print-access-token
```

### Issue: Port Already in Use

**Problem**: Port 8501 or 8081 already in use

**Solution**:
```bash
# Use custom port
python src/streamlit/launcher.py -p 8502
python src/web/launcher.py -p 8082

# Or find and kill process using port
lsof -ti:8501 | xargs kill -9  # On Linux/macOS
```

### Issue: Permission Errors

**Problem**: Permission denied when installing packages

**Solutions**:
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix virtual environment permissions
sudo chown -R $USER:$USER .venv/
```

### Issue: SSL Certificate Errors

**Problem**: SSL errors when making API calls

**Solutions**:
```bash
# Update certificates (macOS)
/Applications/Python\ 3.x/Install\ Certificates.command

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ca-certificates

# Or set environment variable
export SSL_VERIFY=false  # Not recommended for production
```

## 🔧 Advanced Installation Options

### Development Installation

For contributors and developers:

```bash
# Clone with development dependencies
git clone <repository-url>
cd multi-agent-research-platform

# Install with development extras
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Production Installation

For production deployments:

```bash
# Use production requirements
pip install -r requirements-prod.txt

# Set production environment
export ENVIRONMENT=production
export GOOGLE_GENAI_USE_VERTEXAI=true

# Configure logging
export LOG_LEVEL=WARNING
export LOG_FILE=logs/production.log
```

### Custom Installation

For specific use cases:

```bash
# Minimal installation (no web interfaces)
pip install -r requirements-minimal.txt

# Analytics-focused installation
pip install -r requirements.txt plotly pandas numpy

# Development with additional tools
pip install -r requirements.txt jupyter black mypy
```

## 📁 Directory Structure After Installation

```
multi-agent-research-platform/
├── .venv/                    # Virtual environment (if using venv)
├── .env                      # Environment configuration
├── src/
│   ├── agents/              # Agent system
│   ├── config/              # Configuration
│   ├── logging/             # Logging system
│   ├── services/            # Core services
│   ├── web/                 # Web debug interface
│   └── streamlit/           # Streamlit interface
├── multitool_agent/
│   └── .env                 # Agent-specific config
├── docs/                    # Documentation
├── logs/                    # Log files (created at runtime)
├── sessions.db              # SQLite database (created at runtime)
└── requirements.txt         # Dependencies
```

## 🎯 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](QUICKSTART.md)** to learn basic usage
2. **Review the [Configuration Guide](CONFIGURATION.md)** for advanced settings
3. **Explore the [Agent Documentation](AGENTS.md)** to understand agent capabilities
4. **Try the [Streamlit Interface](STREAMLIT_INTERFACE.md)** for user-friendly interactions
5. **Check the [Web Interface Guide](WEB_INTERFACE.md)** for debugging and monitoring

## 💡 Tips for Success

- **Start with the Streamlit interface** for easiest onboarding
- **Use development mode** while learning the platform
- **Enable debug logging** to understand what's happening
- **Check the troubleshooting guide** if you encounter issues
- **Join the community discussions** for help and tips

---

**Need Help?** Check our [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.