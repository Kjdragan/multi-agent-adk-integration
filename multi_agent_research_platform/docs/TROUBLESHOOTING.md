# Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with the Multi-Agent Research Platform.

## 🔍 Quick Diagnosis

### Health Check Commands

```bash
# Check overall system health
curl http://localhost:8080/health

# Detailed system status
curl http://localhost:8080/status

# Check specific API endpoints
curl http://localhost:8080/api/v1/agents
curl http://localhost:8080/api/v1/orchestration/status
```

### Log Analysis

```bash
# View application logs
tail -f logs/app.log

# Check system logs (systemd)
sudo journalctl -u multi-agent-platform -f

# Docker logs
docker logs -f container_name

# Filter by log level
grep "ERROR" logs/app.log
grep "WARNING" logs/app.log
```

## 🚨 Common Issues and Solutions

### 1. Application Won't Start

#### Issue: Port Already in Use
**Symptoms**:
```
Error: [Errno 98] Address already in use
```

**Diagnosis**:
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8080
# or
sudo lsof -i :8080
```

**Solutions**:
```bash
# Option 1: Kill the process using the port
sudo kill -9 <PID>

# Option 2: Use a different port
PORT=8081 python main.py

# Option 3: Change port in configuration
# Edit .env file: PORT=8081
```

#### Issue: Missing Environment Variables
**Symptoms**:
```
KeyError: 'GOOGLE_API_KEY'
Configuration validation error
```

**Diagnosis**:
```bash
# Check environment variables
env | grep GOOGLE
env | grep OPENWEATHER

# Verify .env file exists
ls -la .env
cat .env
```

**Solutions**:
```bash
# Create .env file from template
cp .env.example .env

# Add required variables
echo "GOOGLE_API_KEY=your_key_here" >> .env
echo "OPENWEATHER_API_KEY=your_key_here" >> .env

# Source environment variables
source .env
```

#### Issue: Python/Dependency Problems
**Symptoms**:
```
ModuleNotFoundError: No module named 'fastapi'
ImportError: cannot import name 'get_fast_api_app'
```

**Diagnosis**:
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep fastapi
uv pip list | grep google-adk

# Check virtual environment
which python
echo $VIRTUAL_ENV
```

**Solutions**:
```bash
# Reinstall dependencies
uv sync --force

# Alternative: use pip
pip install -r requirements.txt --force-reinstall

# Clear cache and reinstall
uv cache clean
uv sync

# Create new virtual environment
python -m venv new_env
source new_env/bin/activate
uv sync
```

### 2. Agent Creation Issues

#### Issue: API Key Authentication Errors
**Symptoms**:
```
401 Unauthorized
Invalid API key
Authentication failed with Google AI
```

**Diagnosis**:
```bash
# Test API key directly
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     https://generativelanguage.googleapis.com/v1/models

# Check environment variable
echo $GOOGLE_API_KEY
printenv | grep GOOGLE
```

**Solutions**:
```bash
# Verify API key format
# Gemini API keys start with "AI..."
echo $GOOGLE_API_KEY | head -c 10

# Test with different key
export GOOGLE_API_KEY="your_new_key_here"

# Check quota and billing
# Visit: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com

# Switch between API types
# For local development:
export GOOGLE_GENAI_USE_VERTEXAI=false

# For cloud deployment:
export GOOGLE_GENAI_USE_VERTEXAI=true
```

#### Issue: Agent Registration Failures
**Symptoms**:
```
Agent creation failed
Agent not found in registry
Registry initialization error
```

**Diagnosis**:
```python
# Check agent registry status
from src.agents import AgentRegistry
status = AgentRegistry.get_registry_status()
print(status)

# List registered agents
agents = AgentRegistry.list_agents()
print(f"Registered agents: {len(agents)}")
```

**Solutions**:
```python
# Clear and reinitialize registry
AgentRegistry._agents.clear()
AgentRegistry.initialize()

# Create agents manually
from src.agents import AgentFactory
factory = AgentFactory()
agent = factory.create_llm_agent(role="researcher")
```

### 3. Task Execution Problems

#### Issue: Tasks Timeout or Hang
**Symptoms**:
```
Task execution timeout
Agent not responding
Orchestration timeout
```

**Diagnosis**:
```bash
# Check active tasks
curl http://localhost:8080/api/v1/orchestration/status

# Monitor resource usage
top -p $(pgrep -f "python main.py")
htop

# Check network connectivity
ping google.com
curl -I https://api.openai.com
```

**Solutions**:
```python
# Increase timeout settings
TASK_TIMEOUT_SECONDS = 120
AGENT_TIMEOUT_SECONDS = 60

# Use simpler orchestration strategy
strategy = "single_best"  # instead of "consensus"

# Reduce concurrent tasks
MAX_CONCURRENT_TASKS = 3

# Check agent health before task assignment
agent_status = agent.get_status()
if agent_status["is_healthy"]:
    # Proceed with task
```

#### Issue: Poor Response Quality
**Symptoms**:
```
Irrelevant responses
Incomplete answers
Agent returning empty results
```

**Diagnosis**:
```python
# Check agent configuration
agent_config = agent.get_config()
print(f"Temperature: {agent_config['temperature']}")
print(f"Max tokens: {agent_config['max_tokens']}")

# Review task requirements
print(f"Task: {task}")
print(f"Strategy: {strategy}")
print(f"Required capabilities: {requirements}")
```

**Solutions**:
```python
# Adjust agent parameters
config = {
    "temperature": 0.3,  # Lower for factual tasks
    "max_tokens": 8000,  # Higher for detailed responses
    "model": "gemini-2.5-flash"  # Use appropriate model
}

# Use more specific prompts
task = "Research the environmental impact of renewable energy vs fossil fuels. Include statistics, recent studies, and policy implications."

# Try different orchestration strategies
strategies = ["consensus", "parallel_all", "competitive"]
for strategy in strategies:
    result = await orchestrator.orchestrate_task(task, strategy)
    if result.success and result.quality_score > 0.8:
        break
```

### 4. Web Interface Issues

#### Issue: Streamlit Interface Not Loading
**Symptoms**:
```
Streamlit app not accessible
Connection refused on port 8501
Streamlit command not found
```

**Diagnosis**:
```bash
# Check Streamlit installation
streamlit version

# Check if port is available
netstat -tulpn | grep :8501

# Test Streamlit directly
streamlit hello
```

**Solutions**:
```bash
# Install/reinstall Streamlit
pip install streamlit --upgrade

# Use different port
streamlit run src/streamlit/main.py --server.port 8502

# Check firewall settings
sudo ufw status
sudo ufw allow 8501

# Run with custom launcher
python src/streamlit/launcher.py -p 8502
```

#### Issue: WebSocket Connection Failures
**Symptoms**:
```
WebSocket connection failed
Real-time updates not working
Connection dropped
```

**Diagnosis**:
```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8081/ws');
ws.onopen = () => console.log('Connected');
ws.onerror = (error) => console.log('Error:', error);

# Server-side check
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: test" \
     http://localhost:8081/ws
```

**Solutions**:
```python
# Check WebSocket configuration
ENABLE_WEBSOCKET_MONITORING = True
WEBSOCKET_HEARTBEAT_INTERVAL = 30

# Verify CORS settings
CORS_ORIGINS = ["http://localhost:8501", "http://localhost:3000"]

# Use polling fallback
USE_WEBSOCKET_FALLBACK = True
POLLING_INTERVAL_SECONDS = 5
```

### 5. Performance Issues

#### Issue: Slow Response Times
**Symptoms**:
```
High latency in API responses
Slow agent execution
UI freezing or lag
```

**Diagnosis**:
```bash
# Monitor system resources
top
htop
free -h
df -h

# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/v1/agents

# Analyze performance logs
grep "execution_time" logs/app.log | tail -20
```

**Solutions**:
```python
# Enable caching
ENABLE_RESPONSE_CACHING = True
CACHE_TTL_SECONDS = 300

# Optimize database
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;

# Reduce concurrent tasks
MAX_CONCURRENT_AGENTS = 3
AGENT_POOL_SIZE = 5

# Use connection pooling
CONNECTION_POOL_SIZE = 10
CONNECTION_TIMEOUT = 30
```

#### Issue: High Memory Usage
**Symptoms**:
```
Memory exhaustion
OOM (Out of Memory) errors
System becoming unresponsive
```

**Diagnosis**:
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck python main.py

# Monitor Python memory
python -m memory_profiler main.py
```

**Solutions**:
```python
# Implement memory limits
import resource
resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, -1))  # 2GB limit

# Clear agent memory periodically
if len(agent.memory_cache) > 1000:
    agent.memory_cache.clear()

# Use memory-efficient data structures
from collections import deque
recent_tasks = deque(maxlen=100)

# Enable garbage collection
import gc
gc.collect()
```

### 6. Configuration Issues

#### Issue: Environment-Specific Problems
**Symptoms**:
```
Different behavior in development vs production
Configuration not loading
Environment variables not recognized
```

**Diagnosis**:
```bash
# Check current environment
echo $ENVIRONMENT
python -c "import os; print(os.getenv('ENVIRONMENT', 'not_set'))"

# Verify configuration loading
python -c "
from src.core.config import get_config
config = get_config()
print(config)
"

# Compare environment files
diff .env.development .env.production
```

**Solutions**:
```bash
# Set explicit environment
export ENVIRONMENT=production

# Use environment-specific config files
python main.py --config production.yml

# Override specific settings
export LOG_LEVEL=DEBUG
export GOOGLE_GENAI_USE_VERTEXAI=true
```

### 7. Deployment Issues

#### Issue: Docker Container Problems
**Symptoms**:
```
Container won't start
Build failures
Container exits immediately
```

**Diagnosis**:
```bash
# Check container logs
docker logs container_name

# Inspect container
docker inspect container_name

# Debug interactively
docker run -it --entrypoint /bin/bash image_name

# Check image layers
docker history image_name
```

**Solutions**:
```dockerfile
# Fix common Dockerfile issues
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Use non-root user
RUN useradd -m appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD curl -f http://localhost:8080/health || exit 1
```

#### Issue: Cloud Deployment Failures
**Symptoms**:
```
Cloud Run deployment failed
Build timeout
Service not reachable
```

**Diagnosis**:
```bash
# Check Cloud Build logs
gcloud builds list --limit=5

# Check Cloud Run service
gcloud run services describe SERVICE_NAME --region=REGION

# Test locally with production settings
export GOOGLE_GENAI_USE_VERTEXAI=true
python main.py
```

**Solutions**:
```bash
# Increase build timeout
gcloud config set builds/timeout 1200

# Use cloudbuild.yaml for complex builds
# cloudbuild.yaml
steps:
- name: 'python:3.11'
  entrypoint: 'pip'
  args: ['install', '-r', 'requirements.txt']
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/app', '.']

# Set appropriate resource limits
gcloud run deploy SERVICE_NAME \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10
```

## 🧪 Debugging Tools

### Built-in Debug Endpoints

```bash
# Agent inspection
curl -X POST http://localhost:8081/api/v1/debug/inspect \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_123", "include_performance": true}'

# Performance profiling
curl http://localhost:8081/api/v1/debug/performance

# Log streaming
curl http://localhost:8081/api/v1/debug/logs?level=ERROR&limit=50
```

### Debug Mode Activation

```python
# Enable debug mode in code
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export LOG_LEVEL=DEBUG
export ENABLE_DEBUG_INTERFACE=true

# Run with debug interface
python src/web/launcher.py -e debug
```

### Performance Profiling

```python
# Profile function execution
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    result = orchestrator.orchestrate_task(task, strategy)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

## 📊 Monitoring and Alerts

### Health Monitoring Setup

```python
# Custom health checks
class HealthMonitor:
    def check_agents(self):
        active_agents = AgentRegistry.get_active_count()
        return active_agents > 0
    
    def check_api_keys(self):
        return bool(os.getenv('GOOGLE_API_KEY'))
    
    def check_memory_usage(self):
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 90
```

### Alert Configuration

```yaml
# alerts.yml
alerts:
  high_error_rate:
    threshold: 0.1  # 10% error rate
    window: "5m"
    action: "email"
  
  high_memory_usage:
    threshold: 0.9  # 90% memory usage
    window: "2m"
    action: "restart"
  
  agent_failure:
    threshold: 3  # 3 consecutive failures
    window: "1m"
    action: "deactivate_agent"
```

## 🔧 Recovery Procedures

### Service Recovery

```bash
# Restart application service
sudo systemctl restart multi-agent-platform

# Restart with clean state
sudo systemctl stop multi-agent-platform
rm -f sessions.db
sudo systemctl start multi-agent-platform

# Force restart Docker container
docker restart container_name

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Database Recovery

```bash
# Backup current database
cp sessions.db sessions_backup_$(date +%Y%m%d_%H%M%S).db

# Restore from backup
cp /backups/sessions_latest.db sessions.db

# Reinitialize empty database
rm sessions.db
python -c "
from src.core.database import init_database
init_database()
"
```

### Configuration Recovery

```bash
# Reset configuration to defaults
cp .env.example .env

# Restore from version control
git checkout HEAD -- .env.template
cp .env.template .env

# Regenerate configuration
python scripts/generate_config.py --environment production
```

## 🆘 Getting Help

### Log Collection for Support

```bash
# Collect comprehensive logs
mkdir support_logs
cp logs/*.log support_logs/
docker logs container_name > support_logs/docker.log 2>&1
curl http://localhost:8080/status > support_logs/status.json
env | grep -E "(GOOGLE|OPENWEATHER|ENVIRONMENT)" > support_logs/env_vars.txt

# Create support bundle
tar -czf support_bundle_$(date +%Y%m%d_%H%M%S).tar.gz support_logs/
```

### Diagnostic Information

```python
# Generate diagnostic report
import sys
import platform
import pkg_resources

diagnostic_info = {
    "platform": platform.platform(),
    "python_version": sys.version,
    "packages": [f"{pkg.project_name}=={pkg.version}" 
                for pkg in pkg_resources.working_set],
    "environment": os.environ.copy(),
    "system_health": get_system_health(),
    "agent_status": AgentRegistry.get_registry_status()
}

with open("diagnostic_report.json", "w") as f:
    json.dump(diagnostic_info, f, indent=2, default=str)
```

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Working code samples and tutorials
- **Community Forum**: Discussions and Q&A

---

This troubleshooting guide covers the most common issues you may encounter. For additional help, please check the examples documentation or reach out to the community.