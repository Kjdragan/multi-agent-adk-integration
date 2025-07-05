# Troubleshooting Guide (Current Implementation)

This comprehensive troubleshooting guide helps diagnose and resolve issues with the Multi-Agent Research Platform based on the **actual current implementation**.

## 🔍 Quick Diagnosis

### System Health Check Commands

```bash
# Check overall platform health
curl http://localhost:8081/health

# Detailed system status with services
curl http://localhost:8081/api/v1/status

# Agent registry status
curl http://localhost:8081/api/v1/agents/registry/status

# Service health checks
curl http://localhost:8081/api/v1/services/health
```

### Log Analysis (Run-Based Logging)

```bash
# View latest run logs
ls -la logs/runs/ | tail -5

# Check current run logs
tail -f logs/runs/*/info.log

# Check for errors in latest run
grep "ERROR" logs/runs/*/error.log | tail -10

# View agent performance metrics
cat logs/runs/*/performance.json | jq '.agent_performance'

# Monitor real-time events
tail -f logs/runs/*/events.jsonl | jq '.'
```

## 🚨 Common Issues and Solutions

### 1. Platform Startup Issues

#### Issue: Application Won't Start

**Symptoms**:
```
ImportError: No module named 'agents'
ModuleNotFoundError: No module named 'src.config'
```

**Diagnosis**:
```bash
# Check virtual environment
which python  # Should point to .venv/bin/python

# Check project structure
ls -la src/  # Should show agents/, config/, services/, etc.

# Check PYTHONPATH
echo $PYTHONPATH
```

**Solutions**:
```bash
# Option 1: Activate virtual environment
source .venv/bin/activate  # or venv/bin/activate

# Option 2: Reinstall dependencies
uv sync --force
# OR
pip install -r requirements.txt --force-reinstall

# Option 3: Fix PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

# Option 4: Run from project root
cd /path/to/multi-agent-research-platform
python src/web/launcher.py
```

#### Issue: Port Already in Use

**Symptoms**:
```
OSError: [Errno 98] Address already in use
uvicorn.error: [Errno 98] Address already in use
```

**Diagnosis**:
```bash
# Check what's using the ports
lsof -ti:8081  # Web interface
lsof -ti:8501  # Streamlit interface
netstat -tulpn | grep -E ":808[01]|:8501"
```

**Solutions**:
```bash
# Option 1: Kill processes using ports
lsof -ti:8081 | xargs kill -9
lsof -ti:8501 | xargs kill -9

# Option 2: Use different ports
python src/web/launcher.py -p 8082
python src/streamlit/launcher.py -p 8502

# Option 3: Find and stop background processes
pkill -f "launcher.py"
```

#### Issue: Configuration Validation Errors

**Symptoms**:
```
pydantic.ValidationError: X validation errors
Configuration loading failed
```

**Diagnosis**:
```bash
# Test configuration loading
python -c "
from src.config import get_config
try:
    config = get_config()
    print('Configuration valid')
    print(f'Environment: {config.environment}')
except Exception as e:
    print(f'Configuration error: {e}')
"

# Check environment variables
env | grep -E "GOOGLE|ENVIRONMENT"
```

**Solutions**:
```bash
# Option 1: Reset configuration
cp .env.example .env
# Then edit with your API keys

# Option 2: Fix specific validation issues
export ENVIRONMENT=development
export GOOGLE_GENAI_USE_VERTEXAI=false

# Option 3: Test minimal configuration
cat > .env << 'EOF'
GOOGLE_API_KEY=your_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=false
ENVIRONMENT=development
EOF
```

### 2. Agent System Issues

#### Issue: Agent Creation Failures

**Symptoms**:
```
Agent creation failed
Agent not found in registry
Registry initialization error
```

**Diagnosis**:
```bash
# Check agent registry status
python -c "
from src.agents import AgentRegistry
status = AgentRegistry.get_registry_status()
print('Registry status:', status)
"

# Test basic agent creation
python -c "
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole
factory = AgentFactory()
agent = factory.create_llm_agent(role=LLMRole.RESEARCHER)
print(f'Agent created: {agent.name}')
"
```

**Solutions**:
```bash
# Option 1: Clear and reinitialize registry
python -c "
from src.agents import AgentRegistry
AgentRegistry.clear()
print('Registry cleared')
"

# Option 2: Check service dependencies
python -c "
from src.services import create_development_services
services = create_development_services()
print('Services created successfully')
"

# Option 3: Test with minimal configuration
python -c "
from src.agents.llm_agent import LLMAgent, LLMAgentConfig, LLMRole
config = LLMAgentConfig(role=LLMRole.GENERALIST)
agent = LLMAgent(config)
print(f'Minimal agent created: {agent.agent_id}')
"
```

#### Issue: Google API Authentication Errors

**Symptoms**:
```
401 Unauthorized
403 Forbidden
Invalid API key
Authentication failed with Google AI
```

**Diagnosis**:
```bash
# Check API key format
echo $GOOGLE_API_KEY | head -c 10  # Should start with "AI"

# Test API key directly
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
try:
    models = list(genai.list_models())
    print(f'API key works! Found {len(models)} models')
except Exception as e:
    print(f'API key error: {e}')
"

# Check environment variable loading
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('GOOGLE_API_KEY loaded:', bool(os.getenv('GOOGLE_API_KEY')))
print('USE_VERTEXAI:', os.getenv('GOOGLE_GENAI_USE_VERTEXAI'))
"
```

**Solutions**:
```bash
# Option 1: Verify API key format and permissions
# Get new key from https://makersuite.google.com/app/apikey

# Option 2: Test with direct configuration
export GOOGLE_API_KEY="your_actual_api_key_here"
export GOOGLE_GENAI_USE_VERTEXAI=false

# Option 3: Check quotas and billing
# Visit: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com

# Option 4: Test minimal agent execution
python -c "
import os
os.environ['GOOGLE_API_KEY'] = 'your_key_here'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'false'
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole
factory = AgentFactory()
agent = factory.create_llm_agent(role=LLMRole.GENERALIST)
# Test without actual execution
print('Agent setup successful')
"
```

### 3. Task Execution Problems

#### Issue: Tasks Timeout or Hang

**Symptoms**:
```
Task execution timeout
Agent not responding  
Orchestration timeout after X seconds
```

**Diagnosis**:
```bash
# Check active tasks
curl http://localhost:8081/api/v1/orchestration/active

# Monitor system resources
top -p $(pgrep -f "launcher.py")
ps aux | grep python | grep -E "(web|streamlit)"

# Check network connectivity to external services
ping google.com
curl -I https://generativelanguage.googleapis.com
```

**Solutions**:
```bash
# Option 1: Increase timeout settings
export DEFAULT_TIMEOUT_SECONDS=600
export AGENT_TIMEOUT_SECONDS=120

# Option 2: Use simpler orchestration
# In task execution, use "single_best" instead of "consensus"

# Option 3: Reduce task complexity
# Break complex tasks into smaller parts

# Option 4: Check for deadlocks in logs
grep -i "deadlock\|timeout\|hanging" logs/runs/*/error.log

# Option 5: Restart with clean state
pkill -f launcher.py
rm -f sessions.db  # Clear session state
python src/web/launcher.py -e development
```

#### Issue: Poor Response Quality or Empty Results

**Symptoms**:
```
Agent returning empty results
Irrelevant responses
Error: No suitable agents found
```

**Diagnosis**:
```bash
# Check agent capabilities and registry
python -c "
from src.agents import AgentRegistry
status = AgentRegistry.get_registry_status()
print('Total agents:', status['total_agents'])
print('Agents by capability:', status['agents_by_capability'])
"

# Test agent execution with simple task
curl -X POST http://localhost:8081/api/v1/orchestration/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is 2+2?",
    "strategy": "single_best",
    "priority": "medium"
  }'

# Check recent task results
cat logs/runs/*/events.jsonl | grep task_execution | tail -5 | jq '.'
```

**Solutions**:
```bash
# Option 1: Create agents with proper configuration
python -c "
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole
factory = AgentFactory()

# Create diverse agent team
researcher = factory.create_llm_agent(role=LLMRole.RESEARCHER)
analyst = factory.create_llm_agent(role=LLMRole.ANALYST)
generalist = factory.create_llm_agent(role=LLMRole.GENERALIST)

print('Created agents:', [a.name for a in [researcher, analyst, generalist]])
"

# Option 2: Adjust model configuration
export AUTO_OPTIMIZE_MODELS=true
export ENABLE_THINKING_BUDGETS=true
export PRIORITY_COST=false  # Use better models

# Option 3: Test with known good task
curl -X POST http://localhost:8081/api/v1/orchestration/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain what artificial intelligence is in simple terms",
    "strategy": "adaptive",
    "priority": "medium"
  }'
```

### 4. Interface and Communication Issues

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
python -c "import streamlit; print('Streamlit installed')"
streamlit version

# Check if Streamlit process is running
ps aux | grep streamlit

# Check port availability
netstat -tulpn | grep :8501
```

**Solutions**:
```bash
# Option 1: Install/reinstall Streamlit
pip install streamlit --upgrade

# Option 2: Use launcher script
python src/streamlit/launcher.py -e development

# Option 3: Use different port
python src/streamlit/launcher.py -p 8502

# Option 4: Direct Streamlit execution
cd src/streamlit && streamlit run app.py --server.port 8501

# Option 5: Check firewall settings
sudo ufw status
sudo ufw allow 8501
```

#### Issue: WebSocket Connection Failures

**Symptoms**:
```
WebSocket connection failed
Real-time updates not working
Connection dropped intermittently
```

**Diagnosis**:
```bash
# Test WebSocket connection
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  http://localhost:8081/ws

# Check WebSocket handler
grep -i websocket logs/runs/*/debug.log | tail -10
```

**Solutions**:
```bash
# Option 1: Enable WebSocket monitoring
export ENABLE_WEBSOCKET_MONITORING=true
export WEBSOCKET_HEARTBEAT_INTERVAL=30

# Option 2: Configure CORS properly
export CORS_ORIGINS='["http://localhost:8501", "http://localhost:3000"]'

# Option 3: Use polling fallback
export USE_WEBSOCKET_FALLBACK=true
export POLLING_INTERVAL_SECONDS=5

# Option 4: Restart with debug logging
python src/web/launcher.py -e debug --log-level DEBUG
```

### 5. Performance and Resource Issues

#### Issue: High Memory Usage or Memory Leaks

**Symptoms**:
```
Memory exhaustion
Process killed (OOM)
System becoming unresponsive
```

**Diagnosis**:
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head -10

# Check Python memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Check for memory leaks in logs
grep -i "memory\|leak\|cleanup" logs/runs/*/debug.log
```

**Solutions**:
```bash
# Option 1: Configure memory limits
export MAX_CONCURRENT_AGENTS=3
export MAX_COMPLETED_TASKS_HISTORY=100
export CACHE_TTL_SECONDS=300

# Option 2: Enable automatic cleanup
export ENABLE_AUTOMATIC_CLEANUP=true
export CLEANUP_INTERVAL_MINUTES=15

# Option 3: Use memory-efficient backends
export SESSION_SERVICE_BACKEND=inmemory
export MEMORY_SERVICE_BACKEND=inmemory
export ARTIFACT_SERVICE_BACKEND=inmemory

# Option 4: Set system memory limits
ulimit -v 2097152  # 2GB virtual memory limit

# Option 5: Restart with resource monitoring
python src/web/launcher.py -e debug --enable-resource-monitoring
```

#### Issue: Slow Response Times

**Symptoms**:
```
High latency in API responses
Slow agent execution
UI freezing or lag
```

**Diagnosis**:
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8081/health
# curl-format.txt:
# time_total: %{time_total}\n

# Monitor performance metrics
cat logs/runs/*/performance.json | jq '.execution_metrics'

# Check system load
uptime
iostat 1 5
```

**Solutions**:
```bash
# Option 1: Enable performance optimizations
export ENABLE_CACHING=true
export CACHE_TTL_SECONDS=300
export PRIORITY_SPEED=true

# Option 2: Optimize database performance
export SESSION_SERVICE_BACKEND=database
# For SQLite optimization:
sqlite3 sessions.db "PRAGMA journal_mode = WAL;"
sqlite3 sessions.db "PRAGMA synchronous = NORMAL;"

# Option 3: Reduce concurrent operations
export MAX_CONCURRENT_AGENTS=2
export MAX_CONCURRENT_TASKS=5

# Option 4: Use faster models
export DEFAULT_MODEL=gemini-2.5-flash-lite
export ENABLE_COST_OPTIMIZATION=true
```

### 6. Service Integration Issues

#### Issue: MCP Server Connection Failures

**Symptoms**:
```
MCP server not available
Perplexity/Tavily/Brave API errors
External service timeout
```

**Diagnosis**:
```bash
# Test MCP server availability
python -c "
from src.mcp.orchestrator import MCPOrchestrator
orchestrator = MCPOrchestrator()
print('Available servers:', list(orchestrator.available_servers.keys()))
"

# Check API keys
env | grep -E "PERPLEXITY|TAVILY|BRAVE"

# Test external connectivity
curl -I https://api.perplexity.ai
curl -I https://api.tavily.com
```

**Solutions**:
```bash
# Option 1: Configure MCP server credentials
export PERPLEXITY_API_KEY=your_perplexity_key
export TAVILY_API_KEY=your_tavily_key
export BRAVE_API_KEY=your_brave_key

# Option 2: Disable problematic MCP servers
export ENABLE_MCP_SERVERS=false
# Or disable specific servers:
export DISABLE_PERPLEXITY=true

# Option 3: Use fallback search strategy
export DEFAULT_SEARCH_STRATEGY=single_best
export FALLBACK_TO_GOOGLE_SEARCH=true

# Option 4: Test with basic orchestration
python -c "
from src.agents import AgentOrchestrator
orchestrator = AgentOrchestrator()
# Test without MCP servers
print('Orchestrator created successfully')
"
```

#### Issue: Database Connection Problems

**Symptoms**:
```
Database connection failed
SQLite database locked
Session service unavailable
```

**Diagnosis**:
```bash
# Check database file permissions
ls -la sessions.db

# Test database connectivity
python -c "
from src.services import DatabaseSessionService
try:
    service = DatabaseSessionService()
    print('Database connection successful')
except Exception as e:
    print(f'Database error: {e}')
"

# Check for database locks
lsof sessions.db
```

**Solutions**:
```bash
# Option 1: Reset database
rm -f sessions.db
python -c "
from src.services import DatabaseSessionService
service = DatabaseSessionService()
print('Database recreated')
"

# Option 2: Fix permissions
chmod 664 sessions.db
chown $USER:$USER sessions.db

# Option 3: Use alternative backend
export SESSION_SERVICE_BACKEND=inmemory
export MEMORY_SERVICE_BACKEND=inmemory

# Option 4: Configure database properly
python -c "
import sqlite3
conn = sqlite3.connect('sessions.db')
conn.execute('PRAGMA journal_mode = WAL;')
conn.execute('PRAGMA busy_timeout = 30000;')
conn.close()
print('Database optimized')
"
```

## 🧪 Advanced Debugging

### Enable Debug Mode

```bash
# Start with comprehensive debugging
export LOG_LEVEL=DEBUG
export ENABLE_PERFORMANCE_TRACKING=true
export ENABLE_DEBUG_INTERFACE=true

python src/web/launcher.py -e debug --reload
```

### Debug Agent Execution

```bash
# Test individual agent creation and execution
python -c "
import asyncio
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole

async def test_agent():
    factory = AgentFactory()
    agent = factory.create_llm_agent(role=LLMRole.GENERALIST)
    print(f'Agent created: {agent.name}')
    print(f'Capabilities: {[c.value for c in agent.get_capabilities()]}')
    print(f'Status: {agent.get_status()}')

asyncio.run(test_agent())
"
```

### Profile Performance

```bash
# Use built-in performance tracking
python -c "
from src.platform_logging import create_run_logger
logger = create_run_logger()

with logger.performance_context('test_operation'):
    import time
    time.sleep(1)
    print('Performance tracking enabled')

print(f'Check performance logs in: {logger.run_dir}')
"
```

### Inspect System State

```bash
# Get comprehensive system information
curl -s http://localhost:8081/api/v1/debug/system | jq '.'

# Get agent registry detailed status
curl -s http://localhost:8081/api/v1/agents/registry/detailed | jq '.'

# Get service health with details
curl -s http://localhost:8081/api/v1/services/health?detailed=true | jq '.'
```

## 📊 Monitoring and Health Checks

### Automated Health Monitoring

```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
echo "=== Multi-Agent Platform Health Check ==="
echo "Timestamp: $(date)"

# Check services
curl -s http://localhost:8081/health | jq '.status' || echo "Web service DOWN"
curl -s http://localhost:8501 > /dev/null && echo "Streamlit service UP" || echo "Streamlit service DOWN"

# Check agent registry
AGENTS=$(curl -s http://localhost:8081/api/v1/agents/registry/status | jq '.total_agents')
echo "Total agents: $AGENTS"

# Check recent errors
ERROR_COUNT=$(grep -c "ERROR" logs/runs/*/error.log 2>/dev/null | tail -1 || echo "0")
echo "Recent errors: $ERROR_COUNT"

echo "=== End Health Check ==="
EOF

chmod +x health_check.sh
./health_check.sh
```

### Performance Monitoring

```bash
# Monitor system metrics
python -c "
import psutil
import json

metrics = {
    'cpu_percent': psutil.cpu_percent(interval=1),
    'memory_percent': psutil.virtual_memory().percent,
    'disk_usage': psutil.disk_usage('/').percent,
    'network_connections': len(psutil.net_connections()),
}

print(json.dumps(metrics, indent=2))
"
```

## 🆘 Getting Help

### Collect Diagnostic Information

```bash
# Create comprehensive diagnostic report
cat > collect_diagnostics.sh << 'EOF'
#!/bin/bash
DIAG_DIR="diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p $DIAG_DIR

# System information
python --version > $DIAG_DIR/python_version.txt
pip list > $DIAG_DIR/packages.txt
env | grep -E "(GOOGLE|ENVIRONMENT|LOG)" > $DIAG_DIR/environment.txt

# Configuration
cp .env $DIAG_DIR/env_config.txt 2>/dev/null || echo "No .env file" > $DIAG_DIR/env_config.txt

# Recent logs
cp -r logs/runs/ $DIAG_DIR/ 2>/dev/null || echo "No logs directory" > $DIAG_DIR/no_logs.txt

# System status
curl -s http://localhost:8081/health > $DIAG_DIR/health_status.json 2>/dev/null || echo "Service not running" > $DIAG_DIR/service_status.txt

# Platform status
python -c "
try:
    from src.config import get_config
    config = get_config()
    print('Configuration loaded successfully')
    print(f'Environment: {config.environment}')
except Exception as e:
    print(f'Configuration error: {e}')
" > $DIAG_DIR/config_test.txt 2>&1

echo "Diagnostic information collected in: $DIAG_DIR"
tar -czf ${DIAG_DIR}.tar.gz $DIAG_DIR
echo "Archive created: ${DIAG_DIR}.tar.gz"
EOF

chmod +x collect_diagnostics.sh
./collect_diagnostics.sh
```

### Contact Support

When reporting issues, include:

1. **Platform version and environment**
2. **Complete error messages and stack traces**
3. **Configuration details** (without API keys)
4. **Steps to reproduce the issue**
5. **Diagnostic archive** from the script above

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Complete guides and API reference
- **Examples**: Working code samples and tutorials

---

This troubleshooting guide covers the most common issues with the current implementation. For additional help, check the diagnostic logs in `logs/runs/latest/` for detailed error information and system state.