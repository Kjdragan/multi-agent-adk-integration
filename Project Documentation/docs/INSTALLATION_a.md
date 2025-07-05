# Installation Guide (Current Implementation)

This guide provides comprehensive instructions for installing and setting up the Multi-Agent Research Platform based on the **actual current implementation**.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.11+ recommended for optimal performance)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB+ recommended for multiple agents)
- **Storage**: 3GB free space (includes logs and artifacts)
- **Network**: Internet connection for API calls and MCP server access

### Required API Keys

Before installation, obtain these API keys:

1. **Google AI API Key** (Required)
   - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Free tier available with generous quotas
   - Used for Gemini 2.5 model access

2. **Optional External Service Keys**:
   - **Perplexity API**: For advanced AI-powered research
   - **Tavily API**: For optimized web search
   - **Brave Search API**: For privacy-focused search
   - **OpenWeather API**: For weather-related tools

## 🚀 Installation Methods

### Method 1: UV Package Manager (Recommended)

UV provides faster dependency resolution and installation.

#### Install UV
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv
```

#### Install the Platform
```bash
# Clone repository
git clone <repository-url>
cd multi-agent-research-platform

# Install with UV (creates virtual environment automatically)
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows
```

### Method 2: Traditional pip Installation

```bash
# Clone repository
git clone <repository-url>
cd multi-agent-research-platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Docker Installation (Production)

```bash
# Clone repository
git clone <repository-url>
cd multi-agent-research-platform

# Build Docker image
docker build -t multi-agent-platform .

# Run with environment configuration
docker run -p 8081:8081 -p 8501:8501 --env-file .env multi-agent-platform
```

## 🔧 Configuration Setup

### 1. Environment Configuration

The platform uses environment-specific configuration with Pydantic validation.

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

### 2. Required Environment Variables

Edit your `.env` file with these essential settings:

```bash
# === Google AI Configuration (Required) ===
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id  # Optional for local dev
GOOGLE_CLOUD_LOCATION=us-central1

# === Environment Mode ===
# Local development (recommended for testing)
GOOGLE_GENAI_USE_VERTEXAI=false

# Cloud deployment (for production)
# GOOGLE_GENAI_USE_VERTEXAI=true

# === Application Settings ===
ENVIRONMENT=development  # development, production, demo, minimal
PORT=8081               # Web debug interface port
STREAMLIT_PORT=8501     # Streamlit interface port
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR

# === Optional MCP Service Keys ===
PERPLEXITY_API_KEY=your_perplexity_key    # Optional
TAVILY_API_KEY=your_tavily_key            # Optional
BRAVE_API_KEY=your_brave_search_key       # Optional
OPENWEATHER_API_KEY=your_openweather_key  # Optional

# === Performance Settings ===
MAX_CONCURRENT_AGENTS=5
DEFAULT_TIMEOUT_SECONDS=300
ENABLE_PERFORMANCE_TRACKING=true
ENABLE_CACHING=true
```

### 3. Advanced Configuration Options

```bash
# === Gemini Model Configuration ===
DEFAULT_MODEL=gemini-2.5-flash
ENABLE_THINKING_BUDGETS=true
ENABLE_STRUCTURED_OUTPUT=true
AUTO_OPTIMIZE_MODELS=true

# === Service Backend Selection ===
SESSION_SERVICE_BACKEND=database  # inmemory, database, vertexai
MEMORY_SERVICE_BACKEND=database   # inmemory, database, vertexai
ARTIFACT_SERVICE_BACKEND=local    # inmemory, local, gcs, s3

# === Logging Configuration ===
ENABLE_RUN_BASED_LOGGING=true
LOG_RETENTION_DAYS=30
ENABLE_PERFORMANCE_LOGGING=true

# === Security Settings ===
ENABLE_RATE_LIMITING=true
API_RATE_LIMIT_PER_MINUTE=60
MAX_AGENTS_PER_USER=10
```

## ✅ Verification and Testing

### 1. Verify Installation

```bash
# Check Python and package versions
python --version  # Should be 3.9+
python -c "import google.generativeai; print('Google AI SDK installed')"
python -c "import streamlit; print('Streamlit installed')"
python -c "import fastapi; print('FastAPI installed')"

# Check project structure
ls -la src/  # Should show agents/, config/, services/, etc.
```

### 2. Test Configuration

```bash
# Test environment loading
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Google API Key:', 'Set' if os.getenv('GOOGLE_API_KEY') else 'Missing')
print('Environment:', os.getenv('ENVIRONMENT', 'Not set'))
"

# Test configuration validation
python -c "
from src.config import get_config
config = get_config()
print('Configuration loaded successfully')
print(f'Environment: {config.environment}')
"
```

### 3. Test Agent Creation

```bash
# Test basic agent functionality
python -c "
import sys
sys.path.append('src')
from agents import AgentFactory
from agents.llm_agent import LLMRole

factory = AgentFactory()
agent = factory.create_llm_agent(role=LLMRole.RESEARCHER)
print(f'Agent created: {agent.name}')
print(f'Capabilities: {[cap.value for cap in agent.get_capabilities()]}')
"
```

### 4. Test Service Integration

```bash
# Test service creation
python -c "
from src.services import create_development_services
services = create_development_services()
print('Services created successfully:')
print(f'- Session: {type(services.session_service).__name__}')
print(f'- Memory: {type(services.memory_service).__name__}')
print(f'- Artifact: {type(services.artifact_service).__name__}')
"
```

## 🚀 First Run

### Start the Web Debug Interface

```bash
# Development mode with auto-reload
python src/web/launcher.py -e development --reload

# Production mode
python src/web/launcher.py -e production

# Custom port and host
python src/web/launcher.py -e debug --port 8082 --host 0.0.0.0
```

**Access**: http://localhost:8081

### Start the Streamlit Interface

```bash
# Production interface (recommended for end users)
python src/streamlit/launcher.py

# Development mode with enhanced features
python src/streamlit/launcher.py -e development

# Demo mode with sample data
python src/streamlit/launcher.py -e demo -p 8502
```

**Access**: http://localhost:8501

### Run Both Interfaces Simultaneously

```bash
# Start both interfaces in background
python src/web/launcher.py -e debug &
python src/streamlit/launcher.py -e development &

# Access:
# - Debug Interface: http://localhost:8081
# - Streamlit Interface: http://localhost:8501
```

### Verify Everything Works

1. **Web Debug Interface**: 
   - Load http://localhost:8081
   - Check system health at `/health`
   - Browse API docs at `/docs`

2. **Streamlit Interface**:
   - Load http://localhost:8501
   - Create a test agent through the UI
   - Execute a simple task (e.g., "What is artificial intelligence?")

3. **Test Agent Orchestration**:
   ```bash
   # Quick orchestration test
   curl -X POST http://localhost:8081/api/v1/orchestration/execute \
     -H "Content-Type: application/json" \
     -d '{
       "task": "What are the benefits of renewable energy?",
       "strategy": "adaptive",
       "priority": "medium"
     }'
   ```

## 🐛 Common Installation Issues

### Issue: Import Errors or Missing Modules

**Symptoms**: `ModuleNotFoundError` when running platform

**Solutions**:
```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python or .venv/bin/python

# Reinstall dependencies
uv sync --force
# OR
pip install -r requirements.txt --force-reinstall

# Check PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH
```

### Issue: Google API Authentication Errors

**Symptoms**: `401 Unauthorized` or `403 Forbidden` errors

**Solutions**:
```bash
# Verify API key format (should start with "AI...")
echo $GOOGLE_API_KEY | head -c 10

# Test API key directly
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
models = list(genai.list_models())
print(f'API key works, found {len(models)} models')
"

# Check quotas at https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com
```

### Issue: Port Already in Use

**Symptoms**: `Address already in use` error

**Solutions**:
```bash
# Find process using port
lsof -ti:8081 | xargs kill -9  # Linux/macOS
netstat -ano | findstr :8081   # Windows

# Use different port
python src/web/launcher.py --port 8082
python src/streamlit/launcher.py --port 8502
```

### Issue: Configuration Validation Errors

**Symptoms**: Pydantic validation errors on startup

**Solutions**:
```bash
# Check configuration syntax
python -c "
from src.config import get_config
try:
    config = get_config()
    print('Configuration valid')
except Exception as e:
    print(f'Configuration error: {e}')
"

# Reset to default configuration
cp .env.example .env
# Then edit with your API keys
```

### Issue: Service Initialization Failures

**Symptoms**: Service creation or database errors

**Solutions**:
```bash
# Check database permissions
ls -la sessions.db  # Should be writable

# Recreate database
rm -f sessions.db
python -c "
from src.services import DatabaseSessionService
service = DatabaseSessionService()
print('Database recreated successfully')
"

# Use in-memory services for testing
export SESSION_SERVICE_BACKEND=inmemory
export MEMORY_SERVICE_BACKEND=inmemory
```

## 🔧 Advanced Installation Options

### Development Installation

For contributors and advanced users:

```bash
# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run comprehensive tests
python run_tests.py all

# Install additional development tools
pip install jupyter notebook black mypy
```

### Production Installation

For production deployments:

```bash
# Use production requirements
pip install -r requirements-prod.txt

# Set production environment
export ENVIRONMENT=production
export GOOGLE_GENAI_USE_VERTEXAI=true
export LOG_LEVEL=WARNING

# Configure resource limits
export MAX_CONCURRENT_AGENTS=10
export DEFAULT_TIMEOUT_SECONDS=180
```

### Cloud Deployment (Google Cloud Run)

```bash
# Build and deploy to Cloud Run
gcloud run deploy multi-agent-platform \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true
```

## 📁 Directory Structure After Installation

```
multi-agent-research-platform/
├── .venv/                    # Virtual environment (uv/venv)
├── .env                      # Environment configuration
├── src/
│   ├── agents/              # Agent system (9 LLM roles, orchestration)
│   ├── config/              # Pydantic configuration system  
│   ├── services/            # Multi-backend services
│   ├── mcp/                 # MCP server integrations
│   ├── platform_logging/    # Enterprise logging system
│   ├── streamlit/           # Production user interface
│   ├── web/                 # Debug/monitoring interface
│   ├── tools/               # ADK tool wrappers
│   └── context/             # ADK context management
├── docs/                    # Documentation (this file)
├── logs/                    # Run-based logs (created at runtime)
│   └── runs/                # Per-run log directories
├── sessions.db              # SQLite database (created at runtime)
├── requirements.txt         # Python dependencies
└── run_tests.py            # Test runner script
```

## 🎯 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](QUICKSTART_a.md)** for basic usage
2. **Explore the [Agent Documentation](AGENTS_a.md)** to understand capabilities
3. **Review the [Architecture Guide](ARCHITECTURE_a.md)** for system understanding
4. **Check the [Troubleshooting Guide](TROUBLESHOOTING_a.md)** for common solutions

## 💡 Installation Tips

- **Start with development environment** for learning and testing
- **Use UV for faster dependency management** when available
- **Enable debug logging initially** to understand system behavior
- **Test with simple tasks first** before complex orchestrations
- **Monitor logs directory** for run-based logging structure
- **Check API quotas regularly** to avoid service interruptions

---

**Need Help?** Check the [Troubleshooting Guide](TROUBLESHOOTING_a.md) or review the platform logs in `logs/runs/latest/` for detailed error information.