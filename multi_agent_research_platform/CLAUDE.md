# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies with uv (required)
uv sync

# Configure environment variables
cp .env.template .env
# Edit .env with API keys - CRITICAL: Set GOOGLE_GENAI_USE_VERTEXAI=False for testing
```

### Running the Application
```bash
# ADK Web Interface (debugging/development)
uv run python main.py

# Streamlit Interface (production UX)
uv run streamlit run streamlit_app.py

# Both interfaces simultaneously
uv run python main.py &
uv run streamlit run streamlit_app.py
```

### Testing
```bash
# Recommended: Use test runner (auto-handles PYTHONPATH)
uv run python run_tests.py unit              # Unit tests
uv run python run_tests.py integration       # Integration tests
uv run python run_tests.py all              # All tests
uv run python run_tests.py coverage         # With coverage

# Direct pytest (requires manual PYTHONPATH)
PYTHONPATH=. uv run pytest tests/unit/test_agents.py::TestLLMAgent::test_llm_agent_creation -v

# Single test verification (useful for debugging)
PYTHONPATH=. uv run pytest tests/integration/test_gemini_integration.py -v
```

### Code Quality
```bash
# Formatting
uv run black src/ tests/
uv run isort src/ tests/

# Linting and type checking
uv run ruff check src/ tests/
uv run mypy src/
```

## Architecture Overview

### Multi-Agent System Design
This platform implements a sophisticated multi-agent architecture built on Google ADK v1.5.0 with enterprise-grade orchestration patterns:

**Core Orchestration Flow:**
```
AgentFactory → Agent Selection → TaskAllocation → OrchestrationStrategy → Execution → Results
```

**Key Orchestration Strategies:**
- `SINGLE_BEST`: Optimal agent selection based on capability matching
- `CONSENSUS`: Multiple agents with consensus building 
- `PIPELINE`: Sequential processing through specialized agents
- `COMPETITIVE`: Parallel execution with best result selection
- `ADAPTIVE`: Dynamic strategy selection based on task complexity

### Gemini 2.5 Model Integration
The platform features intelligent model selection with thinking budgets:

```python
# Task complexity drives model selection
TaskComplexity.SIMPLE → Gemini 2.5 Flash-Lite (fastest)
TaskComplexity.MEDIUM → Gemini 2.5 Flash (balanced)
TaskComplexity.COMPLEX → Gemini 2.5 Flash (high budget)
TaskComplexity.CRITICAL → Gemini 2.5 Pro (maximum capability)
```

**Key Configuration:**
- `src/config/gemini_models.py`: Model selection logic and thinking budgets
- `src/agents/factory.py`: Task-optimized agent creation
- Automatic complexity analysis from task descriptions

### Context Management Patterns
The platform enforces proper ADK context usage:

```python
# Required pattern for agent operations
async def execute_task(self, task: str, context: Optional[Dict] = None):
    with InvocationContext() as inv_ctx:
        # Task execution with proper context isolation
        
# Tool operations
async def use_tool(self, tool_name: str, params: Dict):
    with ToolContext(tool_name) as tool_ctx:
        # Tool-specific context management
```

### Service Architecture
**Session Management**: Persistent state with cross-session continuity via `SessionService`
**Memory Service**: Vector embeddings with ADK v1.5.0 `MemoryEntry` structure
**Artifact Service**: Multi-format document processing (PDF, DOCX, images)

### Platform Logging System
Enterprise-grade logging with unique per-run organization:

**Structure:** `logs/runs/TIMESTAMP_INVOCATION-ID/`
- Failure-safe operation (logs persist on crashes)
- LLM-ready JSONL event streams + summary JSON
- Separate files by log level (debug.log, info.log, error.log)
- Performance metrics for agent effectiveness tracking

**Usage:**
```python
from src.platform_logging import RunLogger
logger = RunLogger()  # Auto-creates run directory
logger.info("Agent created", extra={"agent_id": self.agent_id})
```

### MCP Server Integration
Sophisticated MCP (Model Context Protocol) server orchestration for external tools:

**Supported Servers:**
- `PerplexityServer`: AI-powered research
- `TavilyServer`: Web search optimization  
- `BraveServer`: Privacy-focused search
- `OmnisearchServer`: Multi-source aggregation

**Configuration Pattern:**
```python
# MCP servers handle authentication, rate limiting, error recovery
mcp_orchestrator = MCPOrchestrator()
await mcp_orchestrator.execute_tool("perplexity", "search", params)
```

## Critical Configuration

### Authentication Setup
```bash
# Local development/testing (REQUIRED for tests)
GOOGLE_GENAI_USE_VERTEXAI=False
GOOGLE_API_KEY=your_api_key

# Production deployment
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_PROJECT=your_project_id
```

### ADK v1.5.0 Compatibility
The platform has been migrated to ADK v1.5.0 with these critical changes:
- Memory service uses `MemoryEntry` instead of `SearchMemoryResponseEntry`
- Content structure uses `Part` instead of `TextPart`
- ReadonlyContext import from `google.adk.agents.readonly_context`
- Tool configuration uses `ToolRegistry` instead of `ToolsConfig`
- Logging module renamed to `platform_logging` to avoid namespace conflicts

## Agent Development Patterns

### Agent Creation
```python
# Basic agent
from src.agents.factory import AgentFactory
factory = AgentFactory()
agent = factory.create_llm_agent(role=LLMRole.RESEARCHER)

# Task-optimized creation with automatic model selection
agent = factory.create_task_optimized_agent(
    task_description="Complex research analysis task",
    role=LLMRole.RESEARCHER,
    context={"priority": "critical"}
)

# Research team with orchestration
team = factory.create_research_team_for_task(
    task_description="Market analysis research",
    team_size="comprehensive"  # Or "minimal", "standard"
)
```

### Orchestration Usage
```python
from src.agents.orchestrator import AgentOrchestrator, OrchestrationStrategy

orchestrator = AgentOrchestrator()
result = await orchestrator.orchestrate_task(
    task="Research quantum computing applications",
    strategy=OrchestrationStrategy.CONSENSUS,
    requirements=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS]
)
```

## File Structure Understanding

**Critical Directories:**
- `src/agents/`: Core agent implementations, factory, orchestrator
- `src/config/`: Pydantic models, Gemini 2.5 configurations, validation
- `src/services/`: Session/Memory/Artifact services with database integration  
- `src/platform_logging/`: Enterprise logging (renamed from `logging` for namespace safety)
- `src/mcp/`: MCP server integrations with authentication
- `src/tools/`: ADK tool wrappers and custom implementations
- `src/context/`: ADK context management patterns and helpers

**Key Entry Points:**
- `main.py`: ADK Web Interface launcher
- `streamlit_app.py`: Streamlit production interface
- `src/agents/factory.py`: Primary agent creation interface
- `src/agents/orchestrator.py`: Multi-agent coordination engine
- `run_tests.py`: Test execution with automatic PYTHONPATH handling

## Testing Strategy

The platform uses a comprehensive 4-tier testing approach:
- **Unit**: Individual component testing with mocks (fast)
- **Integration**: Component interaction testing (medium speed)  
- **E2E**: Complete workflow testing with real APIs (slow)
- **Performance**: Load testing and scalability validation (very slow)

**Test Execution:** The `run_tests.py` script automatically handles PYTHONPATH and environment setup for consistent test execution across different environments.

## Deployment Architecture

**Development**: Dual interface approach with ADK Web Interface (debugging) + Streamlit (UX)
**Production**: Google Cloud Run deployment with automatic scaling and CORS support
**Container**: Dockerized with proper environment variable injection and health checks

## Data Flow and State Management

### Agent Execution Flow
```
User Request → Agent Selection → Context Creation → Tool Access → Memory Storage → Result Aggregation
```

**State Persistence Pattern:**
- Session state stored in SQLite with cross-session continuity
- Memory embeddings managed through ADK v1.5.0 MemoryEntry structure
- Agent performance metrics tracked for orchestration optimization
- Tool usage history maintained for authentication and rate limiting

### Service Integration Pattern
```python
# All agents follow this dependency injection pattern
class LLMAgent(Agent):
    def __init__(self, config, tools, logger=None, session_service=None, memory_service=None):
        # Services are injected, not created internally
        self.session_service = session_service or SessionService()
        self.memory_service = memory_service or MemoryService()
```

## Critical Code Patterns and Conventions

### Async/Await Consistency
**Rule**: All agent operations must be async to support ADK's event-driven architecture
```python
# CORRECT: Async throughout the chain
async def execute_task(self, task: str) -> AgentResult:
    async with self.session_service.get_session() as session:
        result = await self._process_with_tools(task)
        
# INCORRECT: Mixing sync/async breaks ADK integration
def execute_task(self, task: str) -> AgentResult:  # Missing async
    session = self.session_service.get_session()  # Blocks event loop
```

### Error Handling Pattern
**Rule**: Always use AgentResult for consistent error propagation
```python
try:
    result_data = await self._execute_complex_operation()
    return AgentResult(
        agent_id=self.agent_id,
        success=True,
        result=result_data,
        execution_time_ms=time_taken
    )
except Exception as e:
    self.logger.error(f"Operation failed: {e}")
    return AgentResult(
        agent_id=self.agent_id,
        success=False,
        error=str(e)
    )
```

### Configuration Validation Pattern
**Rule**: All config classes inherit from BaseConfig with Pydantic V2 field validators
```python
from pydantic import BaseModel, Field, field_validator

class AgentConfig(BaseModel):
    name: str = Field(..., min_length=1, description="Agent name")
    
    @field_validator('name')  # V2 style, not @validator
    @classmethod
    def validate_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('Name must be alphanumeric with underscores')
        return v
```

## Testing Infrastructure and Patterns

### Test Fixture Architecture
The platform uses a comprehensive fixture system in `conftest.py`:

```python
# Key fixtures for agent testing
@pytest.fixture
async def agent_factory(app_config, mock_services):
    """Provides pre-configured agent factory with mocked services."""
    
@pytest.fixture
def mock_google_ai_client():
    """Mocks Google AI API calls to avoid rate limits and API costs."""
    
@pytest.fixture
async def test_session(temp_database):
    """Provides isolated database session for each test."""
```

### Mocking Strategy
**Rule**: Mock external APIs but test internal orchestration logic
```python
# CORRECT: Mock external APIs, test orchestration
@patch('src.tools.google_search.GoogleSearchTool.search')
async def test_research_workflow(mock_search, agent_factory):
    mock_search.return_value = {"results": ["test data"]}
    # Test actual orchestration logic
    
# INCORRECT: Mocking internal orchestration defeats the purpose
@patch('src.agents.orchestrator.AgentOrchestrator.orchestrate_task')
async def test_orchestration(mock_orchestrate):  # Tests nothing
```

### API Key Test Management
**Critical**: Tests automatically skip API-dependent tests when keys are missing
```bash
# Tests run with mocks when no API key
pytest tests/integration/  # Skips requires_api_key tests

# Tests run with real APIs when key is set
GOOGLE_API_KEY=real_key pytest tests/integration/  # Runs all tests
```

## Extension and Development Patterns

### Adding New Agents
**Pattern**: Inherit from Agent base class and implement required methods
```python
from src.agents.base import Agent, AgentType, AgentCapability

class NewSpecializedAgent(Agent):
    def __init__(self, config: NewAgentConfig, **kwargs):
        super().__init__(
            agent_type=AgentType.CUSTOM,
            capabilities=[AgentCapability.SPECIALIZED_TASK],
            **kwargs
        )
    
    @abstractmethod
    async def execute_task(self, task: str, context: Optional[Dict] = None) -> AgentResult:
        # Must implement core execution logic
        pass
```

### Adding New MCP Servers
**Pattern**: Extend HTTPMCPServer with authentication and error handling
```python
from src.mcp.base import HTTPMCPServer, MCPServerConfig

class NewMCPServer(HTTPMCPServer):
    def __init__(self, config: NewMCPConfig):
        super().__init__(
            name="new_service",
            base_url=config.api_url,
            auth_config=config.auth
        )
    
    async def execute_operation(self, operation: str, params: Dict) -> Any:
        # Implement with retry logic and rate limiting
        return await self._execute_with_retry(operation, params)
```

### Adding New Tools
**Pattern**: Implement ADK tool interface with proper context management
```python
from google.adk.tools import Tool

class NewCustomTool(Tool):
    def __init__(self, config: ToolConfig):
        super().__init__(name="new_tool", description="Tool description")
        self.config = config
    
    async def execute(self, params: Dict, context: ToolContext) -> Any:
        # Tool implementation with context awareness
        pass
```

## Performance Optimization Strategies

### Agent Performance Monitoring
**Pattern**: Track performance metrics for orchestration optimization
```python
# AgentOrchestrator automatically tracks these metrics:
- Agent response times
- Success/failure rates  
- Resource usage patterns
- Capability effectiveness scores

# Use for intelligent agent selection
best_agent = orchestrator.select_optimal_agent(
    task_type="research",
    performance_threshold=0.8
)
```

### Memory Service Optimization
**Critical**: Memory operations are expensive - use strategic caching
```python
# EFFICIENT: Batch memory operations
memory_entries = await memory_service.batch_store([
    ("context1", metadata1),
    ("context2", metadata2)
])

# INEFFICIENT: Individual memory calls
for item in items:
    await memory_service.store(item)  # Too many round trips
```

### Model Selection Performance
**Strategy**: Use task complexity analysis to optimize model selection
```python
# Automatic optimization based on task analysis
complexity = analyze_task_complexity("Simple question about weather")
# Returns TaskComplexity.SIMPLE → uses fastest Gemini Flash-Lite

complexity = analyze_task_complexity("Comprehensive market analysis...")  
# Returns TaskComplexity.COMPLEX → uses Gemini Pro with thinking budget
```

## Debugging and Troubleshooting Strategies

### Log Analysis Pattern
**Strategy**: Use structured logging for rapid issue identification
```bash
# Check agent execution flow
grep "agent_id.*task_execution" logs/runs/latest/info.log

# Identify performance bottlenecks
grep "execution_time_ms" logs/runs/latest/events.jsonl | jq '.execution_time_ms'

# Track orchestration decisions
grep "orchestration_strategy" logs/runs/latest/debug.log
```

### Common Issue Patterns
**Namespace Conflicts**: Platform logging renamed from `logging` to `platform_logging`
```python
# CORRECT after v1.5.0 migration
from src.platform_logging import RunLogger

# INCORRECT - causes Python logging module conflict
from src.logging import RunLogger  # Breaks standard library
```

**Import Path Issues**: Always use relative imports within platform modules
```python
# CORRECT: Relative imports
from ..services import MemoryService
from ...config import AgentConfig

# INCORRECT: Absolute imports break in different environments  
from src.services import MemoryService  # Fails in test environments
```

**ADK Context Violations**: Always use proper context management
```python
# CORRECT: Proper context isolation
async def execute_task(self, task: str):
    with InvocationContext() as ctx:
        result = await self._process_task(task)
        
# INCORRECT: Missing context causes ADK integration issues
async def execute_task(self, task: str):
    result = await self._process_task(task)  # No context isolation
```

## Security and Authentication Patterns

### API Key Management
**Pattern**: Environment-based key management with fallback chains
```python
# Configuration priority order:
1. Environment variables (GOOGLE_API_KEY)
2. .env file settings  
3. ADC for production (GOOGLE_GENAI_USE_VERTEXAI=True)
4. Default test mocks

# Never hardcode keys in configuration classes
```

### MCP Server Authentication
**Pattern**: Secure credential storage with automatic refresh
```python
class MCPAuthConfig:
    def __init__(self):
        self.api_key = SecretStr(os.getenv("MCP_API_KEY"))
        self.refresh_token = SecretStr(os.getenv("MCP_REFRESH_TOKEN"))
    
    async def get_valid_token(self) -> str:
        # Automatic token refresh logic
        if self._token_expired():
            await self._refresh_token()
        return self.api_key.get_secret_value()
```

## Configuration Management Deep Dive

### Environment-Specific Configurations
**Pattern**: Layered configuration with validation
```python
# Base configuration with validation
class BaseConfig(BaseModel):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True
    )

# Environment-specific overrides
class DevelopmentConfig(BaseConfig):
    debug: bool = True
    log_level: str = "DEBUG"
    
class ProductionConfig(BaseConfig):
    debug: bool = False
    log_level: str = "INFO"
```

### Service Discovery Pattern
**Pattern**: Services register themselves with the platform
```python
# Services auto-register their capabilities
class MemoryService(BaseService):
    def __init__(self):
        super().__init__()
        ServiceRegistry.register(
            service_type="memory",
            instance=self,
            capabilities=["search", "store", "embed"]
        )
```

## Agent Registry and Lifecycle Management

### Agent Registration System
The platform uses a centralized `AgentRegistry` for tracking all agents:

```python
# Automatic registration on agent creation
class Agent(ABC):
    def __init__(self, ...):
        # Agent registers itself automatically
        AgentRegistry.register(self)

# Registry provides powerful querying capabilities
research_agents = AgentRegistry.get_agents_by_capability(AgentCapability.RESEARCH)
capable_agents = AgentRegistry.find_capable_agents([AgentCapability.RESEARCH, AgentCapability.ANALYSIS])
```

### Agent Lifecycle States
**Critical Pattern**: Agents have explicit lifecycle management
```python
# Agent states tracked automatically
class Agent:
    is_active: bool = False
    total_tasks_completed: int = 0
    last_task_time: Optional[datetime] = None

# Lifecycle management
await agent.activate()    # Prepare agent for work
await agent.execute_task(task)  # Increments task counter
await agent.deactivate()  # Clean shutdown
```

### Registry Query Patterns
```python
# Find optimal agent for task
agents = AgentRegistry.find_capable_agents(
    required_capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
    agent_type=AgentType.LLM
)

# Registry status monitoring
status = AgentRegistry.get_registry_status()
# Returns: total_agents, agents_by_type, agents_by_capability, active_agents
```

## Memory Service Architecture Deep Dive

### Multiple Memory Backend Strategy
The platform supports three memory implementations with different use cases:

**InMemoryMemoryService**: Development and testing
- Simple keyword-based search
- No external dependencies
- Perfect for unit tests and rapid development

**DatabaseMemoryService**: Production single-instance deployments
- SQLite with FTS5 full-text search
- Persistent storage across restarts
- Automatic cleanup based on retention policies

**VertexAIRagMemoryService**: Production distributed deployments
- Semantic search with Vertex AI RAG
- Scalable vector embeddings
- Falls back to InMemory if Vertex AI unavailable

### Memory Ingestion Criteria
**Critical**: Not all sessions become memories. Ingestion rules:
```python
def _should_ingest_session(self, session: Session) -> bool:
    criteria = self.config.ingestion_criteria
    
    # Minimum quality thresholds
    if len(session.events) < criteria.get('min_events', 5): return False
    if duration < criteria.get('min_duration_seconds', 30): return False
    if criteria.get('exclude_error_sessions', True) and error_events: return False
    
    return True
```

### Memory Search Scoring
```python
# DatabaseMemoryService uses FTS5 BM25 scoring
# InMemoryMemoryService uses keyword frequency
# VertexAIRagMemoryService uses semantic similarity

# All return SearchMemoryResponse with MemoryEntry objects
results = await memory_service.search_memory(app_name, user_id, query)
for memory in results.memories:
    content = memory.content.parts[0].text  # Extract text content
```

## Test Infrastructure Architecture

### Fixture Dependency Chain
The test system uses a sophisticated fixture hierarchy:
```
event_loop (session)
├── test_config (session)
├── app_config (session) 
│   ├── agent_factory (function)
│   │   ├── test_agent (function)
│   │   └── agent_orchestrator (function)
│   └── mock_services (function)
└── test_session_service (function)
```

### Mock Strategy Layers
**Layer 1**: External API Mocks (always active)
```python
@pytest.fixture
def mock_google_ai_client():
    # Mocks genai.GenerativeModel to avoid API costs
    
@pytest.fixture  
def mock_openweather_client():
    # Mocks requests.get for weather API calls
```

**Layer 2**: Service Mocks (conditional)
```python
# Tests can choose real vs mock services
if requires_real_apis:
    agent = factory.create_llm_agent()  # Uses real Google AI
else:
    agent = factory.create_llm_agent()  # Uses mocked client
```

**Layer 3**: Component Mocks (test-specific)
```python
# Individual tests mock specific components
@patch('src.agents.orchestrator.AgentOrchestrator._select_agents')
async def test_orchestration(mock_select):
    mock_select.return_value = test_team
```

### Test Execution Environments
```python
# Test markers control execution
@pytest.mark.unit          # Fast, isolated tests
@pytest.mark.integration   # Service interaction tests
@pytest.mark.e2e          # Full workflow tests
@pytest.mark.requires_api_key("GOOGLE_API_KEY")  # Conditional on API keys
```

## Migration History and Lessons Learned

### ADK v1.5.0 Breaking Changes Handled
1. **Memory API Changes**: `SearchMemoryResponseEntry` → `MemoryEntry`
2. **Content Structure**: `TextPart` → `Part`
3. **Context Imports**: `google.adk.agents` → `google.adk.agents.readonly_context`
4. **Tool Configuration**: `ToolsConfig` → `ToolRegistry`
5. **Web Framework**: Removed `get_fast_api_app()`, use custom FastAPI

### Migration Strategy That Worked
1. **Systematic Import Updates**: Used grep to find all affected imports
2. **Test-Driven Validation**: Fixed tests first to validate changes
3. **Namespace Conflict Resolution**: Renamed `src/logging` → `src/platform_logging`
4. **Incremental Verification**: Fixed one issue at a time
5. **Documentation Updates**: Maintained migration documentation

### Critical Migration Insights
```python
# Pattern: Always use relative imports within platform
# CORRECT:
from ..platform_logging import RunLogger
from ...config import AgentConfig

# INCORRECT (breaks in different environments):
from src.platform_logging import RunLogger
```

## Service Discovery and Dependency Injection

### Service Registration Pattern
```python
# All services inherit from BaseService and auto-register
class BaseService(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        ServiceRegistry.register(self)

# Services can be discovered by type
memory_service = ServiceRegistry.get_service("memory")
session_service = ServiceRegistry.get_service("session")
```

### Dependency Injection in Agents
```python
# Agents use constructor injection with optional services
class LLMAgent(Agent):
    def __init__(self, 
                 config: LLMAgentConfig,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None):
        
        # Services injected or created with defaults
        self.session_service = session_service or SessionService()
        self.memory_service = memory_service or MemoryService()
```

### Factory Service Wiring
```python
# AgentFactory handles service wiring automatically
class AgentFactory:
    def __init__(self, logger=None, session_service=None, memory_service=None):
        self.shared_services = {
            'logger': logger or RunLogger(),
            'session_service': session_service or SessionService(),
            'memory_service': memory_service or MemoryService()
        }
    
    def create_llm_agent(self, **kwargs):
        # Services automatically injected into all created agents
        return LLMAgent(**kwargs, **self.shared_services)
```

## Event System and Platform Logging

### Run-Based Logging Architecture
**Unique Feature**: Each execution creates isolated log directory
```
logs/runs/TIMESTAMP_INVOCATION-ID/
├── events.jsonl      # Machine-readable event stream
├── summary.json      # Run metadata and final status
├── debug.log         # Debug level messages
├── info.log          # Info level messages
├── error.log         # Error level messages
└── performance.json  # Performance metrics
```

### Structured Event Logging
```python
# RunLogger creates structured events
logger = RunLogger()  # Auto-creates unique run directory

# Events include context automatically
logger.info("Agent created", extra={
    "agent_id": self.agent_id,
    "agent_type": self.agent_type.value,
    "run_id": logger.run_id,
    "timestamp": datetime.utcnow().isoformat()
})

# Performance tracking built-in
with logger.performance_context("task_execution"):
    result = await agent.execute_task(task)
    # Automatically logs execution time and resource usage
```

### Log Analysis Patterns
```bash
# Query recent runs
ls -la logs/runs/ | head -10

# Analyze performance trends
grep "execution_time_ms" logs/runs/*/events.jsonl | jq '.execution_time_ms'

# Find error patterns
grep -r "ERROR" logs/runs/*/error.log | tail -20

# Track agent performance
grep "agent_id.*task_completion" logs/runs/*/events.jsonl
```

## Database Schema and Data Models

### Session Storage Schema
```sql
-- SQLite schema for session persistence
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL,  -- JSON serialized state
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_app_user ON sessions (app_name, user_id);
```

### Memory Storage Schema
```sql
-- Memory entries with full-text search
CREATE TABLE memory_entries (
    id TEXT PRIMARY KEY,
    app_name TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,  -- 'event_text', 'function_call', 'session_state'
    author TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    keywords TEXT NOT NULL,  -- JSON array of extracted keywords
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 virtual table for semantic search
CREATE VIRTUAL TABLE memory_fts 
USING fts5(content, tokenize = 'unicode61 remove_diacritics 2');
```

### Configuration Data Model
```python
# All configs inherit from BaseModel with validation
class BaseConfig(BaseModel):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        validate_default=True
    )

# Hierarchical config structure
class AppConfig(BaseConfig):
    agents: AgentConfig
    services: ServiceConfig
    logging: LoggingConfig
    security: SecurityConfig
```

## Error Recovery and Resilience Patterns

### Agent Error Handling
```python
# Standard error handling pattern
async def execute_task(self, task: str) -> AgentResult:
    try:
        result = await self._perform_task_logic(task)
        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            result=result,
            execution_time_ms=execution_time
        )
    except Exception as e:
        self.logger.error(f"Task execution failed: {e}", extra={
            "task": task,
            "agent_id": self.agent_id,
            "error_type": type(e).__name__
        })
        return AgentResult(
            agent_id=self.agent_id,
            success=False,
            error=str(e)
        )
```

### MCP Server Resilience
```python
# Automatic retry with exponential backoff
class HTTPMCPServer:
    async def _execute_with_retry(self, operation: str, params: Dict) -> Any:
        for attempt in range(self.max_retries):
            try:
                return await self._execute_operation(operation, params)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
```

### Service Health Monitoring
```python
# All services implement health checks
class BaseService:
    async def health_check(self) -> tuple[bool, Dict[str, Any]]:
        try:
            return await self._health_check_impl()
        except Exception as e:
            return False, {"error": str(e), "timestamp": time.time()}

# Health check aggregation
async def system_health_check():
    services = ServiceRegistry.get_all_services()
    health_results = {}
    
    for service in services:
        is_healthy, details = await service.health_check()
        health_results[service.name] = {"healthy": is_healthy, **details}
    
    return health_results
```

## Performance Monitoring and Optimization

### Agent Performance Tracking
```python
# Automatic performance metrics collection
class AgentOrchestrator:
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
    
    async def orchestrate_task(self, task: str) -> OrchestrationResult:
        with self.performance_tracker.track_operation("orchestration"):
            # Track agent selection time
            agents = await self._select_agents(task)
            
            # Track individual agent performance
            results = []
            for agent in agents:
                with self.performance_tracker.track_agent_execution(agent.agent_id):
                    result = await agent.execute_task(task)
                    results.append(result)
            
            return self._aggregate_results(results)
```

### Memory Performance Optimization
```python
# Batch operations for efficiency
class MemoryService:
    async def batch_store(self, entries: List[Tuple[str, Dict]]) -> List[str]:
        """Store multiple entries in single transaction."""
        
    async def batch_search(self, queries: List[str]) -> List[SearchMemoryResponse]:
        """Execute multiple searches efficiently."""

# Usage pattern for optimal performance
memory_results = await memory_service.batch_search([
    "research quantum computing",
    "analyze market trends", 
    "summarize findings"
])
```

### Model Selection Performance
```python
# Automatic model selection based on task complexity
from src.config.gemini_models import analyze_task_complexity, get_optimal_model

complexity = analyze_task_complexity(task_description)
model_config = get_optimal_model(complexity)

# Performance tracking per model
class ModelPerformanceTracker:
    def track_model_usage(self, model_name: str, tokens_used: int, execution_time: float):
        # Updates model performance metrics for future optimization
```

## Development Workflow Best Practices

### Code Organization Principles
1. **Service-Oriented Architecture**: Each major feature is a service
2. **Dependency Injection**: Services don't create dependencies
3. **Interface Segregation**: Small, focused interfaces
4. **Configuration-Driven**: Behavior controlled via config files
5. **Test-First Development**: Write tests before implementation

### Git Workflow Patterns
```bash
# Feature development workflow
git checkout -b feature/new-agent-type

# Make changes with atomic commits
git add src/agents/new_agent.py
git commit -m "feat: add new specialized agent type"

git add tests/unit/test_new_agent.py
git commit -m "test: add unit tests for new agent type"

git add docs/AGENT_TYPES.md
git commit -m "docs: document new agent type capabilities"

# Integration testing
python run_tests.py integration

# Create pull request
gh pr create --title "Add new specialized agent type" --body "Implements X capability"
```

### Code Review Checklist
- [ ] All new code has corresponding tests
- [ ] Configuration follows Pydantic V2 patterns
- [ ] Error handling uses AgentResult pattern
- [ ] Logging includes structured context
- [ ] Memory operations use batch patterns when possible
- [ ] ADK context management is properly implemented
- [ ] Import paths use relative imports within platform
- [ ] Documentation updated for new features

### Debugging Workflow
```python
# Enable debug logging
os.environ["LOG_LEVEL"] = "DEBUG"

# Use performance context for timing
with logger.performance_context("debug_operation"):
    result = await problematic_operation()

# Check recent logs
latest_run = sorted(os.listdir("logs/runs"))[-1]
with open(f"logs/runs/{latest_run}/debug.log") as f:
    debug_logs = f.read()
```

This comprehensive documentation provides the deep architectural understanding, practical patterns, migration insights, and development workflows that would enable a fresh Claude instance to quickly become productive with this sophisticated multi-agent platform.