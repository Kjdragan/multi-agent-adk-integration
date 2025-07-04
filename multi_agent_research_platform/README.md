# Multi-Agent Research Platform

🚀 **Enterprise-grade multi-agent research platform** built on Google ADK v1.5.0 with advanced Gemini 2.5 integration, sophisticated orchestration, and comprehensive monitoring capabilities.

[![Platform](https://img.shields.io/badge/Platform-Google%20ADK%20v1.5.0-blue)](https://cloud.google.com/adk)
[![Models](https://img.shields.io/badge/Models-Gemini%202.5-green)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🌟 Platform Overview

The Multi-Agent Research Platform is a sophisticated, production-ready system featuring intelligent agent orchestration, external service integration, and enterprise-grade monitoring. Built with modern software architecture principles and optimized for both research workflows and scalable deployments.

### ⚡ Key Capabilities

- **🤖 9 Specialized LLM Agent Roles** with Gemini 2.5 thinking budgets and structured output
- **🎭 9 Orchestration Strategies** with intelligent adaptive selection
- **🌐 External Service Integration** via 4 MCP servers (Perplexity, Tavily, Brave, Omnisearch)
- **🖥️ Dual Interface Approach** - Streamlit for users, Web debug for developers
- **📊 Enterprise Logging** with failure-safe run-based organization
- **⚙️ Multi-Backend Services** supporting in-memory, database, and cloud deployments
- **🧵 Thread-Safe Operations** with performance tracking and resource management

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                             │
│  🎨 Streamlit (Production) • 🔧 Web Debug • 🌐 REST APIs     │
├─────────────────────────────────────────────────────────────────┤
│                  Orchestration Layer                           │
│  🎭 Agent Orchestrator (9 Strategies) • 📋 Task Manager       │
├─────────────────────────────────────────────────────────────────┤
│                     Agent System                               │
│  🤖 LLM Agents (9 roles) • 🔄 Workflow • 🎯 Custom           │
├─────────────────────────────────────────────────────────────────┤
│                   Service Layer                                │
│  💾 Session • 🧠 Memory • 📁 Artifact • 📊 Logging          │
├─────────────────────────────────────────────────────────────────┤
│               Integration & Foundation                          │
│  🔗 Google ADK v1.5.0 • 🌐 MCP Servers • 🔌 APIs           │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (3.11+ recommended)
- **UV package manager** or pip
- **Google AI API Key** ([Get it here](https://makersuite.google.com/app/apikey))
- **4GB+ RAM** (8GB+ recommended for multiple agents)

### ⚡ 5-Minute Setup

```bash
# 1. Clone and install
git clone <repository-url>
cd multi-agent-research-platform
uv sync  # Creates .venv automatically

# 2. Configure environment
cp .env.example .env
# Edit .env with your Google API key:
# GOOGLE_API_KEY=your_gemini_api_key_here

# 3. Launch the platform
python src/streamlit/launcher.py  # Production interface (Port 8501)
# OR
python src/web/launcher.py -e debug  # Debug interface (Port 8081)
```

### 🎯 First Research Task

**Via Streamlit Interface (Recommended):**
1. Open http://localhost:8501
2. Create a **Researcher** agent
3. Execute: *"What are the main benefits of renewable energy?"*
4. Watch real-time orchestration and results

**Via REST API:**
```bash
curl -X POST http://localhost:8081/api/v1/orchestration/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze AI applications in healthcare",
    "strategy": "adaptive",
    "priority": "medium"
  }'
```

## 🤖 Agent System

### 9 Specialized LLM Agent Roles

| Agent Role | Primary Capabilities | Best For |
|------------|---------------------|----------|
| 🔬 **RESEARCHER** | Web search, fact-checking, literature review | Information gathering, source verification |
| 📊 **ANALYST** | Statistical analysis, trend identification | Data interpretation, forecasting |
| 🔗 **SYNTHESIZER** | Information integration, conflict resolution | Multi-source summary, knowledge consolidation |
| 🎯 **CRITIC** | Quality assessment, bias detection | Review, error identification, improvement |
| 📋 **PLANNER** | Task breakdown, resource allocation | Project planning, milestone creation |
| 💬 **COMMUNICATOR** | Clear communication, presentation | Stakeholder engagement, content adaptation |
| 🎨 **CREATIVE** | Innovation, brainstorming, original content | Creative solutions, ideation |
| 🎓 **SPECIALIST** | Deep domain expertise, technical precision | Expert-level analysis, industry insights |
| 🌟 **GENERALIST** | Cross-domain tasks, general assistance | Versatile support, initial assessment |

### 🎭 Orchestration Strategies

```python
# 9 Available Orchestration Strategies
OrchestrationStrategy.ADAPTIVE        # 🧠 Intelligent strategy selection (recommended)
OrchestrationStrategy.SINGLE_BEST     # 🏆 Optimal agent selection
OrchestrationStrategy.PARALLEL_ALL    # ⚡ All capable agents simultaneously
OrchestrationStrategy.CONSENSUS       # 🤝 Multiple agents with consensus building
OrchestrationStrategy.PIPELINE        # 🔄 Sequential processing chain
OrchestrationStrategy.COMPETITIVE     # 🏁 Best result selection
OrchestrationStrategy.SEQUENTIAL      # 📋 Step-by-step execution
OrchestrationStrategy.HIERARCHICAL    # 🏗️ Lead agent coordination
OrchestrationStrategy.COLLABORATIVE   # 🤝 Real-time agent cooperation
```

### 🧠 Advanced Agent Features

- **🧭 Intelligent Model Selection**: Automatic Gemini 2.5 model selection based on task complexity
- **💭 Thinking Budgets**: Enhanced reasoning with configurable thinking time
- **📊 Structured Output**: Schema-based responses for complex analysis
- **🎯 Performance Tracking**: Real-time success rates and execution metrics
- **🧵 Thread-Safe Registry**: Concurrent agent operations with RLock protection

## 🌐 Interface Options

### 🎨 Streamlit Interface (Production UX)
**Target**: Researchers, analysts, business users
**Port**: 8501

**Features:**
- ✅ Intuitive visual agent builder
- ✅ Real-time task progress monitoring  
- ✅ Interactive charts and analytics
- ✅ Export capabilities (JSON, CSV, PDF)
- ✅ Multi-environment support (dev/prod/demo)
- ✅ Light/dark theme switching

```bash
# Start production interface
python src/streamlit/launcher.py -e production

# Development mode with debug features
python src/streamlit/launcher.py -e development
```

### 🔧 Web Debug Interface (Developer Tools)
**Target**: Developers, system administrators
**Port**: 8081

**Features:**
- 🔧 Real-time monitoring dashboards
- 🔧 Agent performance analytics
- 🔧 WebSocket communication testing
- 🔧 API documentation at `/docs`
- 🔧 System health monitoring
- 🔧 Debug tools and log analysis

```bash
# Start debug interface with hot reload
python src/web/launcher.py -e debug --reload

# Production monitoring
python src/web/launcher.py -e production
```

## ⚙️ Service Architecture

### Multi-Backend Service Layer

#### 💾 Session Service (3 Backend Options)
- **InMemory**: Development and testing
- **Database**: Production SQLite with connection pooling
- **VertexAI**: Cloud-native with ADK integration

#### 🧠 Memory Service (3 Backend Options)
- **InMemory**: Basic keyword search
- **Database**: SQLite with FTS5 full-text search
- **VertexAI RAG**: Semantic search with vector embeddings

#### 📁 Artifact Service (4 Backend Options)
- **InMemory**: Development/testing
- **LocalFile**: File system with versioning
- **GCS**: Google Cloud Storage
- **S3**: AWS S3 integration

```bash
# Configure service backends via environment variables
export SESSION_SERVICE_BACKEND=database
export MEMORY_SERVICE_BACKEND=database
export ARTIFACT_SERVICE_BACKEND=local
```

## 🌐 External Service Integration

### MCP Server Integration (4 Services)

| Service | Capability | Use Case |
|---------|------------|----------|
| 🧠 **Perplexity** | AI-powered research | Advanced analysis and insights |
| 🔍 **Tavily** | Optimized web search | Fast, relevant web results |
| 🛡️ **Brave Search** | Privacy-focused search | Secure, private research |
| 🌐 **Omnisearch** | Multi-source aggregation | Comprehensive cross-validation |

### Search Strategies
```python
# 8 Available Search Strategies
"ADAPTIVE"           # 🧠 Intelligent strategy selection
"HYBRID_VALIDATION"  # ✅ Cross-validate across multiple sources
"QUALITY_OPTIMIZED"  # ⭐ Best quality results
"SPEED_OPTIMIZED"    # ⚡ Fastest response
"COST_OPTIMIZED"     # 💰 Minimize API costs
"PARALLEL_ALL"       # 🔄 Query all sources simultaneously
"SEQUENTIAL"         # 📋 Try sources in order
"SINGLE_BEST"        # 🏆 Use best single source
```

## 📊 Enterprise Logging

### Run-Based Logging Architecture

Each execution creates an isolated log directory with comprehensive tracking:

```
logs/runs/TIMESTAMP_INVOCATION-ID/
├── events.jsonl      # 📊 Machine-readable event stream
├── summary.json      # 📋 Run metadata and final status
├── debug.log         # 🔍 Debug level messages
├── info.log          # ℹ️ Info level messages
├── error.log         # ❌ Error level messages
└── performance.json  # ⚡ Performance metrics and timings
```

### Key Features
- **🛡️ Failure-Safe Operation**: Logs persist even on crashes
- **🤖 LLM-Ready Formatting**: Structured JSONL events for AI analysis
- **📈 Performance Tracking**: Built-in metrics collection
- **🔍 Real-time Monitoring**: Live log analysis capabilities

## 📁 Project Structure

```
src/
├── agents/                # 🤖 Multi-agent system core
│   ├── base.py           # Agent registry, capabilities, base classes
│   ├── llm_agent.py      # Gemini 2.5 LLM agents (9 roles)
│   ├── orchestrator.py   # 9 orchestration strategies
│   └── factory.py        # Agent creation and team templates
├── config/               # ⚙️ Pydantic configuration system
│   ├── base.py          # Base config with validation
│   ├── gemini_models.py # Gemini 2.5 model selection
│   └── services.py      # Service configurations
├── services/            # 🔧 Multi-backend service layer
│   ├── session.py      # Session management (3 backends)
│   ├── memory.py       # Memory service (3 backends)
│   └── artifact.py     # Artifact handling (4 backends)
├── mcp/                 # 🌐 MCP server integration
│   ├── orchestrator.py # Multi-source search orchestration
│   └── servers/        # Specific server implementations
├── platform_logging/   # 📊 Enterprise logging system
│   ├── logger.py       # Run-based logger implementation
│   └── handlers.py     # Failure-safe file handlers
├── streamlit/          # 🎨 Production user interface
│   ├── app.py         # Main Streamlit application
│   └── launcher.py    # Environment-aware launcher
└── web/                # 🔧 Debug/monitoring interface
    ├── app.py         # FastAPI application
    ├── api.py         # REST API endpoints
    └── launcher.py    # Web interface launcher
```

## 🔧 Configuration

### Required Environment Variables

```bash
# === Google AI Configuration (Required) ===
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=false  # Local development
GOOGLE_CLOUD_PROJECT=your_project_id  # Cloud deployment

# === Application Settings ===
ENVIRONMENT=development  # development, production, demo, minimal
PORT=8081               # Web debug interface port
STREAMLIT_PORT=8501     # Streamlit interface port
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR

# === Service Backend Selection ===
SESSION_SERVICE_BACKEND=database  # inmemory, database, vertexai
MEMORY_SERVICE_BACKEND=database   # inmemory, database, vertexai
ARTIFACT_SERVICE_BACKEND=local    # inmemory, local, gcs, s3

# === Optional MCP Service Keys ===
PERPLEXITY_API_KEY=your_perplexity_key    # Optional
TAVILY_API_KEY=your_tavily_key            # Optional
BRAVE_API_KEY=your_brave_search_key       # Optional

# === Performance Settings ===
MAX_CONCURRENT_AGENTS=5
DEFAULT_TIMEOUT_SECONDS=300
ENABLE_PERFORMANCE_TRACKING=true
```

## 🧪 Testing

### Comprehensive Test Strategy

```bash
# Fast unit tests with mocks (~30 seconds)
python run_tests.py unit

# Service interaction tests (~2 minutes)
python run_tests.py integration

# Full workflow tests (~5 minutes)
python run_tests.py e2e

# Complete test suite (~8 minutes)
python run_tests.py all

# With coverage reporting
python run_tests.py coverage
```

### Test Categories

- **Unit Tests**: Fast, isolated component testing with comprehensive mocks
- **Integration Tests**: Service interaction and API integration testing
- **E2E Tests**: Complete workflow testing with real agent orchestration
- **Performance Tests**: Load testing and scalability validation

## 📖 Usage Examples

### Basic Agent Creation

```python
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole

# Create agent factory
factory = AgentFactory()

# Create specialized research agent
researcher = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    auto_optimize_model=True,  # Intelligent model selection
    enable_thinking=True,      # Enhanced reasoning
    priority_cost=False        # Quality over cost
)

# Execute research task
result = await researcher.execute_task(
    "Analyze renewable energy market trends in 2024"
)

print(f"Success: {result.success}")
print(f"Result: {result.result}")
print(f"Execution time: {result.execution_time_ms}ms")
```

### Advanced Orchestration

```python
from src.agents.orchestrator import AgentOrchestrator, OrchestrationStrategy
from src.agents.base import AgentCapability

# Create orchestrator
orchestrator = AgentOrchestrator()

# Execute with adaptive strategy selection
result = await orchestrator.orchestrate_task(
    task="Comprehensive analysis of AI applications in healthcare",
    strategy=OrchestrationStrategy.ADAPTIVE,  # Auto-selects best approach
    required_capabilities=[
        AgentCapability.RESEARCH,
        AgentCapability.ANALYSIS,
        AgentCapability.SYNTHESIS
    ]
)

print(f"Strategy used: {result.strategy_used}")
print(f"Agents involved: {len(result.agent_results)}")
print(f"Consensus score: {result.consensus_score}")
```

### Predefined Agent Teams

```python
# Create specialized research team
research_team = factory.create_agent_suite(
    suite_type=AgentSuite.RESEARCH_TEAM
)
# Creates: RESEARCHER + ANALYST + SYNTHESIZER

# Create content development team
content_team = factory.create_agent_suite(
    suite_type=AgentSuite.CONTENT_CREATION
)
# Creates: CREATIVE + COMMUNICATOR + CRITIC
```

## 🚀 Deployment

### Local Development

```bash
# Start both interfaces simultaneously
python src/web/launcher.py -e debug &
python src/streamlit/launcher.py -e development

# Access:
# - Streamlit Interface: http://localhost:8501
# - Web Debug Interface: http://localhost:8081
```

### Production Deployment

#### Google Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: multi-agent-platform
spec:
  template:
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/multi-agent-platform
        env:
        - name: GOOGLE_GENAI_USE_VERTEXAI
          value: "true"
        - name: ENVIRONMENT
          value: "production"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
```

#### Docker

```bash
# Build and run with Docker
docker build -t multi-agent-platform .
docker run -p 8081:8081 -p 8501:8501 --env-file .env multi-agent-platform
```

## 🔍 Monitoring & Health Checks

### System Health Monitoring

```bash
# Check overall platform health
curl http://localhost:8081/health

# Detailed system status with services
curl http://localhost:8081/api/v1/status

# Agent registry status
curl http://localhost:8081/api/v1/agents/registry/status
```

### Performance Analytics

- **📊 Real-time Dashboards**: Live agent activity and performance metrics
- **📈 Success Rate Tracking**: Agent and orchestration effectiveness
- **⏱️ Response Time Monitoring**: Task execution performance
- **🔄 Resource Usage**: Memory, CPU, and API quota tracking

## 🐛 Troubleshooting

### Quick Diagnostics

```bash
# Test configuration
python -c "from src.config import get_config; print('Config OK')"

# Test agent creation
python -c "from src.agents import AgentFactory; print('Agents OK')"

# Test services
python -c "from src.services import create_development_services; print('Services OK')"

# Check API key format
echo $GOOGLE_API_KEY | head -c 10  # Should start with "AI"
```

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **API Key Issues**: Verify Google API key format and quotas
3. **Port Conflicts**: Use alternative ports with `-p` flag
4. **Service Failures**: Check service backend configurations

## 📚 Documentation

Comprehensive documentation is available:

- **[📖 Quick Start Guide](docs/QUICKSTART_a.md)** - Get started in 5 minutes
- **[🤖 Agent Documentation](docs/AGENTS_a.md)** - Complete agent system guide
- **[🏗️ Architecture Overview](docs/ARCHITECTURE_a.md)** - System design and patterns
- **[⚙️ Installation Guide](docs/INSTALLATION_a.md)** - Detailed setup instructions
- **[🔧 Troubleshooting Guide](docs/TROUBLESHOOTING_a.md)** - Common issues and solutions
- **[🔧 Development Guide](CLAUDE.md)** - Developer documentation and patterns

## 🔮 Platform Capabilities

### ✅ Current Features
- **9 LLM Agent Roles** with Gemini 2.5 integration
- **9 Orchestration Strategies** with adaptive selection
- **Multi-Backend Services** (in-memory, database, cloud)
- **Enterprise Logging** with run-based organization  
- **MCP Server Integration** (4 external services)
- **Dual Interface Approach** (Streamlit + Web Debug)
- **Thread-Safe Agent Registry** with performance tracking
- **Comprehensive Testing** (unit, integration, e2e)
- **Intelligent Model Selection** based on task complexity
- **Real-time Monitoring** and health checks

### 🔄 Advanced Features
- **Hot Reload** in development mode
- **Comprehensive Mocking** for external APIs
- **Structured Logging** for debugging
- **Performance Metrics** collection
- **Error Recovery** and resilience patterns
- **Configuration Validation** with Pydantic

## 🤝 Contributing

1. **Follow Established Patterns**: Use the agent and service patterns documented in [CLAUDE.md](CLAUDE.md)
2. **Comprehensive Testing**: Add unit, integration, and e2e tests for new features
3. **Proper Error Handling**: Use `AgentResult` pattern for consistent error propagation
4. **Structured Logging**: Include appropriate logging context for debugging
5. **Documentation Updates**: Update relevant documentation for new features

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:

1. **📖 Check Documentation**: Review [comprehensive guides](docs/) for solutions
2. **🔍 Analyze Logs**: Examine `logs/runs/latest/` for detailed error information
3. **🧪 Run Diagnostics**: Use provided diagnostic commands for system health
4. **🐛 Report Issues**: Open an issue with detailed logs and error information

---

**🚀 Ready to get started?** Follow the [Quick Start Guide](docs/QUICKSTART_a.md) or jump straight into the [5-Minute Setup](#-5-minute-setup) above!