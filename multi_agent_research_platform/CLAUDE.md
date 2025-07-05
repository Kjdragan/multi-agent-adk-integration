# CLAUDE.md - Multi-Agent Research Platform

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the Multi-Agent Research Platform repository, based on **actual source code analysis** and current implementation.

## 🏗️ Platform Overview (Current Reality)

The Multi-Agent Research Platform is an enterprise-grade, sophisticated multi-agent system built on Google ADK v1.5.0 with advanced Gemini 2.5 integration, thread-safe orchestration, external service integration, and comprehensive monitoring capabilities.

### Core System Architecture (Actual Implementation)

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                             │
│  Streamlit (Production) • Web Debug (Development) • REST APIs  │
├─────────────────────────────────────────────────────────────────┤
│                  Orchestration Layer                           │
│  Agent Orchestrator (9 Strategies) • Task Manager • Registry   │
├─────────────────────────────────────────────────────────────────┤
│                     Agent System                               │
│  LLM Agents (9 roles) • Workflow • Custom • Factory           │
├─────────────────────────────────────────────────────────────────┤
│                   Service Layer                                │
│  Session • Memory • Artifact • Logging (Multi-backend)        │
├─────────────────────────────────────────────────────────────────┤
│               Integration & Foundation                          │
│  Google ADK v1.5.0 • MCP Servers • External APIs             │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Development Commands

### Environment Setup
```bash
# Install dependencies with uv (required)
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env with API keys - CRITICAL: Set GOOGLE_GENAI_USE_VERTEXAI=false for local development
```

### Running the Platform
```bash
# Streamlit Interface (Production UX) - Port 8501
uv run python src/streamlit/launcher.py

# Web Debug Interface (Development/Monitoring) - Port 8081  
uv run python src/web/launcher.py -e debug

# Both interfaces simultaneously
uv run python src/web/launcher.py -e debug &
uv run python src/streamlit/launcher.py -e development

# Custom configurations
uv run python src/streamlit/launcher.py -e production -p 8502
uv run python src/web/launcher.py -e development --host 0.0.0.0
```

### Testing Strategy
```bash
# Comprehensive test runner (handles PYTHONPATH automatically)
uv run python run_tests.py unit              # Fast unit tests with mocks
uv run python run_tests.py integration       # Service interaction tests  
uv run python run_tests.py e2e              # Full workflow tests
uv run python run_tests.py all              # Complete test suite
uv run python run_tests.py coverage         # With coverage reporting

# Direct pytest execution (uv automatically handles environment)
uv run pytest tests/unit/test_agents.py::TestLLMAgent -v
uv run pytest tests/integration/test_orchestration.py -v
```

### Code Quality & Development
```bash
# Formatting and linting
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/

# Health checks
curl http://localhost:8081/health
curl http://localhost:8081/api/v1/status
```

## 🤖 Agent System (Actual Implementation)

### Agent Registry (Thread-Safe with RLock)
- **Location**: `src/agents/base.py:AgentRegistry`
- **Thread Safety**: Uses `threading.RLock()` for nested calls and concurrent access
- **Capabilities**: Performance tracking, capability indexing, agent lifecycle management
- **Registry Status**: Real-time monitoring of agent counts, capabilities, and performance

### LLM Agents (9 Specialized Roles with Gemini 2.5)
**Location**: `src/agents/llm_agent.py:LLMAgent`

#### Available Roles and Capabilities Matrix
```python
from src.agents.llm_agent import LLMRole, LLMAgent, LLMAgentConfig

# 9 Specialized LLM Agent Roles:
LLMRole.RESEARCHER     # Web search, fact-checking, literature review
LLMRole.ANALYST        # Statistical analysis, trend identification  
LLMRole.SYNTHESIZER    # Information integration, conflict resolution
LLMRole.CRITIC         # Quality assessment, bias detection
LLMRole.PLANNER        # Task breakdown, resource allocation
LLMRole.COMMUNICATOR   # Clear communication, presentation
LLMRole.CREATIVE       # Innovation, brainstorming, original content
LLMRole.SPECIALIST     # Deep domain expertise, technical precision
LLMRole.GENERALIST     # Cross-domain tasks, general assistance
```

#### Agent Creation Patterns
```python
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole

factory = AgentFactory()

# Basic agent creation
researcher = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    auto_optimize_model=True,
    enable_thinking=True
)

# Advanced configuration
specialist = factory.create_llm_agent(
    role=LLMRole.SPECIALIST,
    model=GeminiModel.PRO,  # Best model for critical tasks
    enable_thinking=True,
    thinking_budget=2000,   # Extended thinking budget
    priority_cost=False     # Quality over cost
)

# Predefined agent teams
research_team = factory.create_agent_suite(
    suite_type=AgentSuite.RESEARCH_TEAM
)
# Creates: RESEARCHER + ANALYST + SYNTHESIZER
```

### Orchestration Strategies (9 Implementations)
**Location**: `src/agents/orchestrator.py:AgentOrchestrator`

```python
from src.agents.orchestrator import AgentOrchestrator, OrchestrationStrategy

orchestrator = AgentOrchestrator()

# 9 Orchestration Strategies Available:
OrchestrationStrategy.ADAPTIVE        # Intelligent strategy selection (recommended)
OrchestrationStrategy.SINGLE_BEST     # Optimal agent selection
OrchestrationStrategy.PARALLEL_ALL    # All capable agents simultaneously  
OrchestrationStrategy.CONSENSUS       # Multiple agents with consensus building
OrchestrationStrategy.PIPELINE        # Sequential processing chain
OrchestrationStrategy.COMPETITIVE     # Best result selection
OrchestrationStrategy.SEQUENTIAL      # Step-by-step execution
OrchestrationStrategy.HIERARCHICAL    # Lead agent coordination
OrchestrationStrategy.COLLABORATIVE   # Real-time agent cooperation

# Usage example
result = await orchestrator.orchestrate_task(
    task="Comprehensive analysis of AI market trends",
    strategy=OrchestrationStrategy.ADAPTIVE,  # Auto-selects best approach
    priority="medium"
)
```

### Agent Capabilities (14 Core Capabilities)
**Location**: `src/agents/base.py:AgentCapability`

```python
from src.agents.base import AgentCapability

# 14 Core Agent Capabilities:
AgentCapability.REASONING            # Logical reasoning and analysis
AgentCapability.RESEARCH             # Information gathering and research
AgentCapability.ANALYSIS             # Data and content analysis
AgentCapability.SYNTHESIS            # Information synthesis and summarization
AgentCapability.PLANNING             # Task planning and decomposition
AgentCapability.EXECUTION            # Task execution and orchestration
AgentCapability.COMMUNICATION        # Agent-to-agent communication
AgentCapability.LEARNING             # Learning from interactions
AgentCapability.TOOL_USE             # Using external tools and APIs
AgentCapability.MEMORY_ACCESS        # Accessing and storing memories
AgentCapability.CONTEXT_MANAGEMENT   # Managing conversation context
AgentCapability.FACT_CHECKING        # Verifying information accuracy
AgentCapability.CONTENT_GENERATION   # Creating new content
AgentCapability.DECISION_MAKING      # Making informed decisions

# Capability queries
capable_agents = AgentRegistry.find_capable_agents([
    AgentCapability.RESEARCH,
    AgentCapability.FACT_CHECKING
])
```

## ⚙️ Service Architecture (Multi-Backend)

### Service Layer Design
**Locations**: `src/services/`

#### Session Service (3 Backend Options)
```python
from src.services import SessionService, DatabaseSessionService, InMemorySessionService

# Development/Testing
session_service = InMemorySessionService()

# Production Single-Instance
session_service = DatabaseSessionService()  # SQLite with connection pooling

# Production Distributed
session_service = VertexAISessionService()  # Cloud-native with ADK integration
```

#### Memory Service (3 Backend Options with Different Search Capabilities)
```python
from src.services import MemoryService, DatabaseMemoryService, InMemoryMemoryService

# Basic keyword search (development)
memory_service = InMemoryMemoryService()

# SQLite with FTS5 full-text search (production)
memory_service = DatabaseMemoryService()  # BM25 ranking, retention policies

# Semantic search with vector embeddings (cloud)
memory_service = VertexAIRagMemoryService()  # Vertex AI RAG integration
```

#### Artifact Service (4 Backend Options)
```python
from src.services import ArtifactService, LocalFileArtifactService

# Local file system with versioning
artifact_service = LocalFileArtifactService()

# Cloud storage options
artifact_service = GCSArtifactService()     # Google Cloud Storage
artifact_service = S3ArtifactService()      # AWS S3
artifact_service = InMemoryArtifactService() # Development/testing
```

### Service Factory Pattern
```python
from src.services import create_development_services, create_production_services

# Development services (in-memory, fast)
services = create_development_services()

# Production services (database, persistent)
services = create_production_services(
    session_backend="database",
    memory_backend="database", 
    artifact_backend="local"
)

# Access services
session_service = services.session_service
memory_service = services.memory_service
artifact_service = services.artifact_service
```

## 📊 Platform Logging System (Enterprise-Grade)

### Run-Based Logging Architecture
**Location**: `src/platform_logging/logger.py:RunLogger`

#### Unique Features
- **Per-Run Isolation**: Each execution creates unique directory `logs/runs/TIMESTAMP_INVOCATION-ID/`
- **Failure-Safe Operation**: Logs persist even on crashes
- **LLM-Ready Formatting**: Structured JSONL events for AI analysis
- **Performance Tracking**: Built-in metrics collection

#### Log Directory Structure
```
logs/runs/TIMESTAMP_INVOCATION-ID/
├── events.jsonl      # Machine-readable event stream
├── summary.json      # Run metadata and final status  
├── debug.log         # Debug level messages
├── info.log          # Info level messages
├── error.log         # Error level messages
└── performance.json  # Performance metrics and timings
```

#### Usage Patterns
```python
from src.platform_logging import create_run_logger

# Auto-creates unique run directory
logger = create_run_logger()

# Structured event logging
logger.info("Agent created", extra={
    "agent_id": agent.agent_id,
    "agent_type": agent.agent_type.value,
    "capabilities": [cap.value for cap in agent.get_capabilities()]
})

# Performance tracking
with logger.performance_context("task_execution"):
    result = await agent.execute_task(task)
    # Automatically logs execution time and resource usage
```

#### Log Analysis Commands
```bash
# View recent runs
ls -la logs/runs/ | tail -5

# Monitor real-time events
tail -f logs/runs/*/events.jsonl | jq '.'

# Check for errors
grep "ERROR" logs/runs/*/error.log | tail -10

# Analyze performance
cat logs/runs/*/performance.json | jq '.agent_performance'
```

## 🌐 Gemini 2.5 Integration (Advanced Features)

### Intelligent Model Selection
**Location**: `src/config/gemini_models.py`

```python
from src.config.gemini_models import analyze_task_complexity, get_optimal_model

# Automatic complexity analysis and model selection
complexity = analyze_task_complexity(task_description)

# Model selection based on complexity:
# TaskComplexity.SIMPLE → Gemini 2.5 Flash-Lite (fastest, cost-effective)
# TaskComplexity.MEDIUM → Gemini 2.5 Flash (balanced performance)
# TaskComplexity.COMPLEX → Gemini 2.5 Flash (high thinking budget)
# TaskComplexity.CRITICAL → Gemini 2.5 Pro (maximum capability)

model_config = get_optimal_model(
    complexity=complexity,
    priority_speed=False,  # False = quality over speed
    priority_cost=False    # False = quality over cost
)
```

### Advanced Gemini Features
```python
# Thinking budgets for enhanced reasoning
agent_config = LLMAgentConfig(
    role=LLMRole.ANALYST,
    enable_thinking=True,
    thinking_budget=2000,  # Extended thinking for complex analysis
    enable_structured_output=True,
    output_schema_type="analysis"
)

# Auto-optimization based on task context
auto_optimized_agent = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    auto_optimize_model=True,  # Automatically selects best model
    enable_thinking=True,
    priority_cost=False  # Use best models regardless of cost
)
```

## 🔗 MCP Server Integration (4 External Services)

### MCP Orchestrator
**Location**: `src/mcp/orchestrator.py:MCPOrchestrator`

#### Available MCP Servers
```python
from src.mcp.orchestrator import MCPOrchestrator

mcp_orchestrator = MCPOrchestrator()

# 4 Integrated External Services:
# 1. Perplexity - AI-powered research and analysis
# 2. Tavily - Optimized web search  
# 3. Brave Search - Privacy-focused search
# 4. Omnisearch - Multi-source aggregation

# Intelligent service routing
result = await mcp_orchestrator.execute_search(
    query="latest AI research developments",
    strategy="HYBRID_VALIDATION",  # Cross-validates across multiple sources
    optimize_for="quality"  # quality, speed, or cost
)
```

#### Search Strategies
```python
# 8 Available Search Strategies:
"SINGLE_BEST"        # Use best single source
"PARALLEL_ALL"       # Query all sources simultaneously
"SEQUENTIAL"         # Try sources in order
"ADAPTIVE"           # Intelligent strategy selection
"HYBRID_VALIDATION"  # Cross-validate results
"COST_OPTIMIZED"     # Minimize API costs
"SPEED_OPTIMIZED"    # Fastest response
"QUALITY_OPTIMIZED"  # Best quality results
```

## 🖥️ Interface Layer (Dual Interface Approach)

### Streamlit Interface (Production UX)
**Location**: `src/streamlit/`
**Target**: End users, researchers, business analysts
**Port**: 8501

#### Features
- **Multiple Environments**: Development, production, demo modes
- **Theme Support**: Light/dark mode switching
- **Interactive Agent Creation**: Visual agent builder
- **Real-time Progress**: Live task execution monitoring
- **Analytics Dashboard**: Performance charts and metrics
- **Export Capabilities**: JSON, CSV, PDF export

#### Usage
```bash
# Production mode (optimized performance)
python src/streamlit/launcher.py -e production

# Development mode (debug features)
python src/streamlit/launcher.py -e development

# Demo mode (sample data)
python src/streamlit/launcher.py -e demo
```

### Web Debug Interface (Development/Monitoring)
**Location**: `src/web/`
**Target**: Developers, system administrators
**Port**: 8081

#### Features
- **Real-time Monitoring**: Live agent and task dashboards
- **WebSocket Communication**: Real-time updates and events
- **Agent Performance Analytics**: Detailed metrics and trends
- **System Health Monitoring**: Service status and diagnostics
- **API Documentation**: Interactive OpenAPI docs at `/docs`
- **Debug Tools**: Agent registry inspection, log analysis

#### Usage
```bash
# Debug mode with enhanced logging
python src/web/launcher.py -e debug --reload

# Production monitoring
python src/web/launcher.py -e production

# Custom configuration
python src/web/launcher.py -e debug --port 8082 --host 0.0.0.0
```

## 🔧 Configuration System (Pydantic-Based)

### Environment-Specific Configurations
**Location**: `src/config/`

#### Required Environment Variables
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

# === Performance Tuning ===
MAX_CONCURRENT_AGENTS=5
DEFAULT_TIMEOUT_SECONDS=300
ENABLE_PERFORMANCE_TRACKING=true
```

#### Configuration Validation
```python
from src.config import get_config

# Test configuration loading
config = get_config()
print(f"Environment: {config.environment}")
print(f"Agents config: {config.agents}")
print(f"Services config: {config.services}")
```

## 🧪 Testing Infrastructure

### Test Execution Strategy
```bash
# Test categories with different execution speeds:
python run_tests.py unit         # Fast tests with mocks (~30 seconds)
python run_tests.py integration  # Service tests (~2 minutes)  
python run_tests.py e2e         # Full workflow tests (~5 minutes)
python run_tests.py all         # Complete suite (~8 minutes)

# API key dependent tests
GOOGLE_API_KEY=real_key python run_tests.py integration  # Uses real APIs
python run_tests.py integration  # Skips API tests, uses mocks
```

### Test Organization
```
tests/
├── unit/               # Fast, isolated component tests
│   ├── test_agents.py         # Agent creation and capabilities
│   ├── test_orchestrator.py   # Orchestration logic
│   └── test_services.py       # Service implementations
├── integration/        # Service interaction tests
│   ├── test_agent_orchestration.py  # Multi-agent workflows
│   ├── test_gemini_integration.py   # Gemini API integration
│   └── test_mcp_integration.py      # MCP server integration
├── e2e/               # End-to-end workflow tests
│   ├── test_research_workflow.py    # Complete research tasks
│   └── test_interface_integration.py # UI integration tests
└── performance/       # Load and performance tests
    └── test_concurrent_agents.py    # Scalability testing
```

### Mock Strategy
```python
# Comprehensive mocking for external dependencies
@pytest.fixture
def mock_google_ai_client():
    """Mocks Google AI API to avoid costs and rate limits"""
    
@pytest.fixture
def mock_mcp_servers():
    """Mocks external MCP servers (Perplexity, Tavily, etc.)"""
    
@pytest.fixture
async def test_services():
    """Provides in-memory services for isolated testing"""
```

## 🚦 Development Workflow

### Agent Development Pattern
```python
# 1. Create agent with proper configuration
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole

factory = AgentFactory()
agent = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    auto_optimize_model=True,
    enable_thinking=True
)

# 2. Execute tasks with proper error handling
try:
    result = await agent.execute_task("Research quantum computing applications")
    if result.success:
        print(f"Success: {result.result}")
        print(f"Execution time: {result.execution_time_ms}ms")
    else:
        print(f"Error: {result.error}")
except Exception as e:
    print(f"Unexpected error: {e}")

# 3. Check performance metrics
metrics = agent.get_performance_metrics()
print(f"Success rate: {metrics['success_rate_percent']}%")
print(f"Average response time: {metrics['average_response_time_ms']}ms")
```

### Orchestration Development Pattern
```python
# 1. Create orchestrator with services
from src.agents.orchestrator import AgentOrchestrator, OrchestrationStrategy

orchestrator = AgentOrchestrator()

# 2. Execute with intelligent strategy selection
result = await orchestrator.orchestrate_task(
    task="Comprehensive market analysis of renewable energy sector",
    strategy=OrchestrationStrategy.ADAPTIVE,  # Auto-selects best approach
    required_capabilities=[
        AgentCapability.RESEARCH,
        AgentCapability.ANALYSIS,
        AgentCapability.SYNTHESIS
    ]
)

# 3. Analyze orchestration results
print(f"Strategy used: {result.strategy_used}")
print(f"Agents involved: {len(result.agent_results)}")
print(f"Consensus score: {result.consensus_score}")
```

### Service Development Pattern
```python
# 1. Create services with appropriate backends
from src.services import create_development_services

services = create_development_services()

# 2. Use services through agents (dependency injection)
agent = factory.create_llm_agent(
    role=LLMRole.ANALYST,
    session_service=services.session_service,
    memory_service=services.memory_service,
    artifact_service=services.artifact_service
)

# 3. Services are automatically integrated
result = await agent.execute_task("Analyze data trends")
# Agent automatically uses provided services for context, memory, and artifacts
```

## 🐛 Debugging and Troubleshooting

### Common Development Issues

#### 1. Import Path Issues
```python
# CORRECT: Use relative imports within platform
from ..services import MemoryService
from ...config import AgentConfig

# INCORRECT: Absolute imports break in test environments
from src.services import MemoryService  # Fails in tests
```

#### 2. Environment Configuration
```bash
# CORRECT: Test configuration loading
python -c "from src.config import get_config; print('Config OK')"

# Check API key format
echo $GOOGLE_API_KEY | head -c 10  # Should start with "AI"

# Verify service creation
python -c "from src.services import create_development_services; print('Services OK')"
```

#### 3. Agent Registry Issues
```python
# CORRECT: Check registry status
from src.agents import AgentRegistry

status = AgentRegistry.get_registry_status()
print(f"Total agents: {status['total_agents']}")
print(f"Agents by capability: {status['agents_by_capability']}")

# Clear registry if needed
AgentRegistry.clear()
```

#### 4. Performance Issues
```bash
# Check system resources
curl http://localhost:8081/api/v1/status

# Monitor agent performance
cat logs/runs/*/performance.json | jq '.agent_performance'

# Check for memory leaks
grep -i "memory" logs/runs/*/debug.log
```

### Log Analysis Patterns
```bash
# Real-time monitoring
tail -f logs/runs/*/events.jsonl | jq '.'

# Performance analysis
grep "execution_time_ms" logs/runs/*/events.jsonl | jq '.execution_time_ms'

# Error investigation
grep "ERROR" logs/runs/*/error.log | tail -20

# Agent activity tracking
grep "agent_id.*task_execution" logs/runs/*/info.log
```

## 🔐 Security and Best Practices

### API Key Management
```bash
# Development (local testing)
GOOGLE_API_KEY=your_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=false

# Production (cloud deployment)
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your_project_id
```

### Error Handling Patterns
```python
# Standard error handling for agents
async def execute_task(self, task: str) -> AgentResult:
    try:
        result = await self._process_task(task)
        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            result=result,
            execution_time_ms=time_taken
        )
    except Exception as e:
        self.logger.error(f"Task failed: {e}", extra={
            "task": task,
            "agent_id": self.agent_id
        })
        return AgentResult(
            agent_id=self.agent_id,
            success=False,
            error=str(e)
        )
```

### Performance Optimization
```python
# Use batch operations for efficiency
memory_results = await memory_service.batch_search([
    "query1", "query2", "query3"
])

# Enable caching for better performance
export ENABLE_CACHING=true
export CACHE_TTL_SECONDS=300

# Optimize for speed vs quality based on use case
agent = factory.create_llm_agent(
    role=LLMRole.GENERALIST,
    priority_speed=True,  # Speed over quality
    priority_cost=True    # Cost over quality
)
```

## 📚 File Structure (Actual Implementation)

### Critical Source Code Locations
```
src/
├── agents/                # Multi-agent system core
│   ├── __init__.py       # Agent exports and imports
│   ├── base.py           # Agent registry, capabilities, base classes
│   ├── llm_agent.py      # Gemini 2.5 LLM agents (9 roles)
│   ├── workflow_agent.py # Multi-step process orchestration
│   ├── custom_agent.py   # Domain-specific specialized agents
│   ├── orchestrator.py   # 9 orchestration strategies
│   └── factory.py        # Agent creation and team templates
├── config/               # Pydantic configuration system
│   ├── __init__.py      # Configuration exports
│   ├── base.py          # Base config with validation
│   ├── app.py           # Application configurations
│   ├── agents.py        # Agent-specific configurations
│   ├── gemini_models.py # Gemini 2.5 model selection
│   └── services.py      # Service configurations
├── services/            # Multi-backend service layer
│   ├── __init__.py     # Service factory and exports
│   ├── session.py      # Session management (3 backends)
│   ├── memory.py       # Memory service (3 backends)
│   ├── artifact.py     # Artifact handling (4 backends)
│   └── factory.py      # Service creation and wiring
├── mcp/                 # MCP server integration
│   ├── __init__.py     # MCP exports
│   ├── base.py         # MCP base classes and interfaces
│   ├── orchestrator.py # Multi-source search orchestration
│   └── servers/        # Specific server implementations
│       ├── perplexity.py # AI-powered research
│       ├── tavily.py     # Web search optimization
│       ├── brave.py      # Privacy-focused search
│       └── omnisearch.py # Multi-source aggregation
├── platform_logging/   # Enterprise logging system
│   ├── __init__.py     # Logging exports
│   ├── logger.py       # Run-based logger implementation
│   ├── handlers.py     # Failure-safe file handlers
│   └── formatters.py   # LLM-ready formatting
├── streamlit/          # Production user interface
│   ├── app.py         # Main Streamlit application
│   ├── components.py  # Reusable UI components
│   └── launcher.py    # Environment-aware launcher
├── web/                # Debug/monitoring interface  
│   ├── app.py         # FastAPI application
│   ├── api.py         # REST API endpoints
│   ├── dashboards.py  # Real-time monitoring
│   └── launcher.py    # Web interface launcher
├── tools/              # ADK tool wrappers
│   ├── google_search.py # Google Search integration
│   └── code_execution.py # Python code execution
└── context/            # ADK context management
    ├── patterns.py    # Context patterns for tools/memory
    └── managers.py    # Context lifecycle management
```

### Key Entry Points
- **`src/streamlit/launcher.py`**: Production user interface
- **`src/web/launcher.py`**: Development/debug interface  
- **`src/agents/factory.py`**: Primary agent creation
- **`src/agents/orchestrator.py`**: Multi-agent coordination
- **`run_tests.py`**: Test execution with environment handling

## 🚀 Quick Start Checklist

### 1. Installation
```bash
git clone <repository-url>
cd multi-agent-research-platform
uv sync
cp .env.example .env
# Edit .env with GOOGLE_API_KEY
```

### 2. Verification
```bash
# Test configuration
python -c "from src.config import get_config; print('Config OK')"

# Test agent creation
python -c "from src.agents import AgentFactory; print('Agents OK')"

# Test services
python -c "from src.services import create_development_services; print('Services OK')"
```

### 3. First Run
```bash
# Start Streamlit interface
python src/streamlit/launcher.py

# Or start debug interface
python src/web/launcher.py -e debug

# Access:
# Streamlit: http://localhost:8501
# Web Debug: http://localhost:8081
```

### 4. First Task
```python
# Via Python API
from src.agents import AgentFactory
from src.agents.llm_agent import LLMRole

factory = AgentFactory()
agent = factory.create_llm_agent(role=LLMRole.RESEARCHER)
result = await agent.execute_task("What are the benefits of renewable energy?")
print(result.result)
```

```bash
# Via REST API
curl -X POST http://localhost:8081/api/v1/orchestration/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What are the benefits of renewable energy?",
    "strategy": "adaptive",
    "priority": "medium"
  }'
```

## 📈 Current Platform Capabilities (Verified)

### ✅ Operational Features
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

### 🔧 Development Features
- **Hot Reload** in development mode
- **Comprehensive Mocking** for external APIs
- **Structured Logging** for debugging
- **Performance Metrics** collection
- **Error Recovery** and resilience patterns
- **Configuration Validation** with Pydantic

This documentation represents the **actual current state** of the Multi-Agent Research Platform, providing accurate information for productive development and debugging.