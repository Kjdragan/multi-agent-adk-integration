# Multi-Agent Research Platform - Manual Testing Guide

This guide walks you through manually testing all the key functionality of the Multi-Agent Research Platform to understand its capabilities and verify everything is working correctly.

## Prerequisites

1. **Environment Setup**
   ```bash
   cd multi_agent_research_platform
   uv sync  # Install dependencies
   ```

2. **Configuration Check**
   ```bash
   # Verify your .env file has the required keys
   cat .env | grep -E "(GOOGLE_API_KEY|GOOGLE_GENAI_USE_VERTEXAI)"
   ```

## 🌐 Testing the Web Interface (Development/Debugging)

### 1. Launch the Web Interface
```bash
uv run --isolated python src/web/launcher.py
```

**Expected Output:**
- Server starts on `http://0.0.0.0:8080`
- Demo agents are created automatically
- Real-time monitoring enabled

### 2. Web Interface Features to Test

#### **Main Dashboard** (`http://localhost:8080`)
- [ ] **Agent Status Panel**: Should show 5 demo agents (Research Specialist, Analysis Specialist, etc.)
- [ ] **Performance Metrics**: Real-time updates every 5 seconds
- [ ] **Task Queue**: Initially empty, ready for new tasks
- [ ] **System Health**: All services should show "healthy" status

#### **API Documentation** (`http://localhost:8080/docs`)
- [ ] **Interactive API Docs**: Swagger UI should load
- [ ] **Available Endpoints**: 
  - `/agents` - Agent management
  - `/tasks` - Task execution
  - `/orchestrate` - Multi-agent orchestration
  - `/health` - System health

#### **Agent Management**
1. **View Agents**: Go to `/agents` endpoint in API docs
2. **Test Agent Creation**: Use the POST `/agents` endpoint
   ```json
   {
     "role": "researcher",
     "name": "Test Agent",
     "model": "gemini-2.5-flash"
   }
   ```
3. **Verify Agent Registry**: Check that new agent appears in dashboard

#### **Task Execution**
1. **Single Agent Task**: Use POST `/tasks/execute`
   ```json
   {
     "agent_id": "research-specialist-xxxxx",
     "task": "What are the latest trends in renewable energy?",
     "context": {}
   }
   ```
2. **Check Response**: Should get structured AgentResult with:
   - `success: true`
   - `result`: Generated content
   - `execution_time_ms`: Performance metrics
   - `metadata`: Additional context

#### **Multi-Agent Orchestration**
1. **Test Orchestration**: Use POST `/orchestrate`
   ```json
   {
     "task": "Research and analyze the impact of AI on healthcare",
     "strategy": "consensus",
     "requirements": ["research", "analysis"]
   }
   ```
2. **Monitor Progress**: Watch the dashboard for:
   - Agent selection process
   - Task distribution
   - Results aggregation

### 3. WebSocket Real-time Updates
- [ ] **Open Browser Console**: Check for WebSocket connection
- [ ] **Real-time Updates**: Agent status should update live
- [ ] **Task Progress**: Watch tasks move through the queue in real-time

---

## 📊 Testing the Streamlit Interface (Production UX)

### 1. Launch Streamlit Interface
```bash
uv run --isolated python src/streamlit/launcher.py
```

**Expected Output:**
- Streamlit app starts on `http://localhost:8501`
- Interactive UI for non-technical users

### 2. Streamlit Features to Test

#### **Agent Team Creation**
1. **Sidebar Navigation**: Check "Create Agents" section
2. **Team Templates**: Test each option:
   - [ ] **Research Team**: Creates researcher, analyst, writer
   - [ ] **Analysis Team**: Creates data analysis specialists  
   - [ ] **Content Team**: Creates content creation specialists
   - [ ] **Custom Setup**: Individual agent configuration

#### **Interactive Task Execution**
1. **Task Input**: Enter a complex research task:
   ```
   "Analyze the current state of quantum computing and its potential impact on cybersecurity. Provide a comprehensive report with market analysis and future predictions."
   ```
2. **Agent Selection**: Choose appropriate agents or team
3. **Execution Monitoring**: Watch progress bars and status updates
4. **Results Display**: Review formatted output with:
   - Executive summary
   - Detailed findings
   - Data visualizations (if applicable)

#### **Model Configuration**
1. **Model Selection**: Test different Gemini models:
   - [ ] **Gemini 2.5 Flash-Lite**: Fast responses
   - [ ] **Gemini 2.5 Flash**: Balanced performance
   - [ ] **Gemini 2.5 Pro**: Maximum capability
2. **Thinking Budget**: Adjust thinking time for complex tasks
3. **Structured Output**: Enable for data extraction tasks

#### **Memory and Context**
1. **Session Continuity**: Ask follow-up questions to previous tasks
2. **Memory Search**: Test "Search Previous Results" feature
3. **Context Building**: Verify agents remember conversation history

---

## 🔧 Testing Core Agent Functionality

### 1. Individual Agent Types

#### **LLM Agents**
```bash
# Test in Python console or create a script
uv run --isolated python -c "
from src.agents.factory import AgentFactory
from src.agents.llm_agent import LLMRole
import asyncio

factory = AgentFactory()

# Test different roles
researcher = factory.create_llm_agent(role=LLMRole.RESEARCHER, name='Test Researcher')
analyst = factory.create_llm_agent(role=LLMRole.ANALYST, name='Test Analyst')
writer = factory.create_llm_agent(role=LLMRole.COMMUNICATOR, name='Test Writer')

async def test_agent(agent, task):
    result = await agent.execute_task(task)
    print(f'{agent.name}: {result.success}')
    if result.success:
        print(f'Result: {result.result[:100]}...')
    return result

# Test tasks
async def main():
    await test_agent(researcher, 'Research electric vehicle market trends')
    await test_agent(analyst, 'Analyze this data: EV sales grew 60% in 2023')
    await test_agent(writer, 'Write a summary of EV market analysis')

asyncio.run(main())
"
```

#### **Custom Agents**
```bash
uv run --isolated python -c "
from src.agents.factory import AgentFactory
from src.agents.custom_agent import CustomAgentType

factory = AgentFactory()
fact_checker = factory.create_custom_agent(
    agent_type=CustomAgentType.FACT_CHECKER,
    name='Test Fact Checker'
)
print(f'Created: {fact_checker.name} with capabilities: {fact_checker.get_capabilities()}')
"
```

#### **Workflow Agents**
```bash
uv run --isolated python -c "
from src.agents.factory import AgentFactory

factory = AgentFactory()
workflow = factory.create_workflow_agent(
    name='Test Workflow',
    description='Multi-step research workflow'
)
print(f'Created: {workflow.name}')
"
```

### 2. Agent Orchestration Strategies

#### **Single Best Strategy**
Test with a simple task to see optimal agent selection:
```python
from src.agents.orchestrator import AgentOrchestrator, OrchestrationStrategy
from src.agents.base import AgentCapability

orchestrator = AgentOrchestrator()
result = await orchestrator.orchestrate_task(
    task="What is the capital of France?",
    strategy=OrchestrationStrategy.SINGLE_BEST,
    requirements=[AgentCapability.RESEARCH]
)
```

#### **Consensus Strategy**
Test with a controversial topic requiring multiple perspectives:
```python
result = await orchestrator.orchestrate_task(
    task="Should AI development be regulated? Provide pros and cons.",
    strategy=OrchestrationStrategy.CONSENSUS,
    requirements=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS]
)
```

#### **Pipeline Strategy**
Test with a multi-step task:
```python
result = await orchestrator.orchestrate_task(
    task="Research renewable energy, analyze market trends, and write executive summary",
    strategy=OrchestrationStrategy.PIPELINE,
    requirements=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS, AgentCapability.COMMUNICATION]
)
```

---

## 💾 Testing Services and Infrastructure

### 1. Session Management
```bash
uv run --isolated python -c "
from src.services.session import SessionService
import asyncio

async def test_sessions():
    service = SessionService()
    await service.start()
    
    # Create session
    session = await service.create_session('test-app', 'user-123')
    print(f'Created session: {session.session_id}')
    
    # Store data
    await service.store_interaction(session.session_id, 'user', 'Hello AI')
    await service.store_interaction(session.session_id, 'assistant', 'Hello! How can I help?')
    
    # Retrieve session
    retrieved = await service.get_session(session.session_id)
    print(f'Session has {len(retrieved.events)} events')
    
    await service.stop()

asyncio.run(test_sessions())
"
```

### 2. Memory Service
```bash
uv run --isolated python -c "
from src.services.memory import MemoryService
import asyncio

async def test_memory():
    service = MemoryService()
    await service.start()
    
    # Store memory
    memory_id = await service.store_memory(
        app_name='test-app',
        session_id='session-123',
        user_id='user-123',
        content='Renewable energy research findings',
        content_type='research_result',
        metadata={'topic': 'renewable_energy', 'quality': 'high'}
    )
    print(f'Stored memory: {memory_id}')
    
    # Search memory
    results = await service.search_memory('test-app', 'user-123', 'renewable energy')
    print(f'Found {len(results.memories)} memories')
    
    await service.stop()

asyncio.run(test_memory())
"
```

### 3. Artifact Service
```bash
uv run --isolated python -c "
from src.services.artifact import ArtifactService
import asyncio

async def test_artifacts():
    service = ArtifactService()
    await service.start()
    
    # Store artifact
    artifact_id = await service.store_artifact(
        'test-report.txt',
        b'This is a test research report content',
        'text/plain'
    )
    print(f'Stored artifact: {artifact_id}')
    
    # Retrieve artifact
    content = await service.get_artifact(artifact_id)
    print(f'Retrieved: {content[:50]}...')
    
    # List artifacts
    artifacts = await service.list_artifacts()
    print(f'Total artifacts: {len(artifacts)}')
    
    await service.stop()

asyncio.run(test_artifacts())
"
```

---

## 🔍 Testing Advanced Features

### 1. Model Selection and Thinking Budgets
```bash
uv run --isolated python -c "
from src.config.gemini_models import analyze_task_complexity, select_model_for_task

# Test complexity analysis
simple_task = 'What is 2+2?'
complex_task = 'Analyze the geopolitical implications of renewable energy adoption on global supply chains and provide strategic recommendations for multinational corporations.'

simple_complexity = analyze_task_complexity(simple_task)
complex_complexity = analyze_task_complexity(complex_task)

print(f'Simple task complexity: {simple_complexity}')
print(f'Complex task complexity: {complex_complexity}')

# Test model selection
simple_model = select_model_for_task(simple_task)
complex_model = select_model_for_task(complex_task)

print(f'Simple task model: {simple_model.name}')
print(f'Complex task model: {complex_model.name}')
"
```

### 2. MCP Server Integration
```bash
uv run --isolated python -c "
from src.mcp.orchestrator import MCPOrchestrator
import asyncio

async def test_mcp():
    orchestrator = MCPOrchestrator()
    
    # List available servers
    servers = orchestrator.list_available_servers()
    print(f'Available MCP servers: {list(servers.keys())}')
    
    # Test if servers are configured
    for server_name in servers:
        try:
            server = orchestrator.get_server(server_name)
            print(f'{server_name}: {server.name} - Ready')
        except Exception as e:
            print(f'{server_name}: Error - {e}')

asyncio.run(test_mcp())
"
```

### 3. Performance Monitoring
Check the logs for performance data:
```bash
# View recent performance logs
ls -la logs/runs/
latest_run=$(ls -t logs/runs/ | head -1)
echo "Latest run: $latest_run"

# Check performance metrics
cat logs/runs/$latest_run/performance.json | jq '.agent_performance'

# View execution logs
tail -20 logs/runs/$latest_run/info.log
```

---

## 🧪 Testing with Different Task Types

### 1. Research Tasks
```
"Investigate the latest developments in quantum computing and their potential applications in cryptography."
```

### 2. Analysis Tasks  
```
"Given this data: 'Global EV sales: 2020: 3.1M, 2021: 6.6M, 2022: 10.5M, 2023: 14.1M', analyze the trends and predict 2024-2025 sales."
```

### 3. Creative Tasks
```
"Write a compelling executive summary for a startup that uses AI to optimize renewable energy grid management."
```

### 4. Multi-step Complex Tasks
```
"Research the current state of carbon capture technology, analyze its economic feasibility, compare different approaches, and provide strategic recommendations for energy companies."
```

### 5. Domain-specific Tasks
```
"Analyze the regulatory landscape for autonomous vehicles in the EU and US, focusing on liability frameworks and safety standards."
```

---

## 📋 Testing Checklist

### Core Functionality
- [ ] Web interface launches and loads dashboard
- [ ] Streamlit interface launches and renders correctly
- [ ] Agents can be created via both interfaces
- [ ] Tasks execute successfully with valid API keys
- [ ] Multi-agent orchestration works
- [ ] Results are properly formatted and displayed

### Service Infrastructure  
- [ ] Session management stores and retrieves data
- [ ] Memory service can store and search content
- [ ] Artifact service handles file storage
- [ ] Logging captures execution details

### Agent Capabilities
- [ ] LLM agents respond appropriately to different roles
- [ ] Custom agents provide specialized functionality
- [ ] Workflow agents can coordinate multi-step processes
- [ ] Agent registry tracks all created agents

### Performance and Monitoring
- [ ] Real-time updates work in web interface
- [ ] Performance metrics are collected
- [ ] Error handling works gracefully
- [ ] Logs provide useful debugging information

### Configuration and Models
- [ ] Different Gemini models can be selected
- [ ] Task complexity analysis works
- [ ] Thinking budgets affect response quality
- [ ] API keys are properly configured

---

## 🚨 Troubleshooting Common Issues

### API Key Issues
```bash
# Check if API key is set
echo $GOOGLE_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     "https://generativelanguage.googleapis.com/v1beta/models"
```

### Service Startup Issues
```bash
# Check for port conflicts
netstat -tulpn | grep :8080
netstat -tulpn | grep :8501

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Database Issues
```bash
# Reset session database
rm -f data/sessions.db

# Reset memory database  
rm -f data/memory.db

# Clear logs
rm -rf logs/runs/*
```

### Performance Issues
```bash
# Monitor resource usage
htop

# Check disk space
df -h

# Monitor network connections
ss -tulpn
```

---

## 📊 Expected Performance Benchmarks

### Response Times
- **Simple tasks**: < 2 seconds
- **Medium tasks**: 5-15 seconds  
- **Complex tasks**: 15-60 seconds
- **Multi-agent tasks**: 30-120 seconds

### Resource Usage
- **Memory**: 200-500 MB baseline
- **CPU**: < 50% during normal operation
- **Disk**: < 10 MB/hour for logs

### Accuracy Expectations
- **Factual questions**: > 95% accuracy
- **Analysis tasks**: High-quality insights
- **Creative tasks**: Coherent, relevant content
- **Multi-step tasks**: Logical progression

---

This comprehensive testing guide should help you thoroughly evaluate the Multi-Agent Research Platform's functionality and verify that all components are working correctly. Start with the web interface for immediate visual feedback, then proceed through the various features systematically.