# Contributing Guide

Welcome to the Multi-Agent Research Platform! This guide provides comprehensive information for developers who want to contribute to the project.

## 🚀 Getting Started

### Prerequisites

**Required Software**:
- Python 3.9 or higher
- UV package manager (preferred) or pip
- Git
- Docker (for testing containers)

**Recommended Tools**:
- VS Code with Python extension
- GitHub CLI
- Pre-commit hooks

### Development Environment Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/multi-agent-research-platform.git
cd multi-agent-research-platform

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Set up environment variables
cp .env.example .env.development
# Edit .env.development with your API keys

# Run tests to verify setup
python -m pytest
```

## 📁 Project Structure

### Core Architecture

```
multi-agent-research-platform/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   │   ├── base.py              # Abstract base classes
│   │   ├── llm_agent.py         # LLM-based agents
│   │   ├── workflow_agent.py    # Workflow orchestration
│   │   ├── custom_agent.py      # Specialized agents
│   │   ├── orchestrator.py      # Multi-agent coordination
│   │   └── factory.py           # Agent creation patterns
│   ├── core/                    # Core infrastructure
│   │   ├── config/              # Configuration management
│   │   ├── logging/             # Logging system
│   │   ├── database/            # Data persistence
│   │   └── services/            # Core services
│   ├── web/                     # Web interface
│   │   ├── config.py           # Web configuration
│   │   ├── interface.py        # Core web interface
│   │   ├── api.py              # REST API endpoints
│   │   ├── handlers.py         # WebSocket handlers
│   │   └── dashboards.py       # Monitoring dashboards
│   ├── streamlit/              # Streamlit interface
│   │   ├── main.py             # Main Streamlit app
│   │   ├── components/         # UI components
│   │   └── pages/              # Application pages
│   └── tools/                  # External tool integrations
├── tests/                      # Test suite
├── docs/                       # Documentation
├── examples/                   # Usage examples
└── scripts/                    # Utility scripts
```

### Module Responsibilities

**`src/agents/`**: All agent-related functionality
- Agent base classes and interfaces
- Specialized agent implementations
- Orchestration and coordination logic
- Agent creation and management

**`src/core/`**: Foundation infrastructure
- Configuration management
- Logging and monitoring
- Database operations
- Service architecture

**`src/web/`**: Web-based interfaces
- REST API endpoints
- WebSocket handling
- Dashboard components
- Debug interfaces

**`src/streamlit/`**: Production user interface
- Interactive Streamlit application
- User-friendly components
- Session management

## 🛠️ Development Guidelines

### Code Style

We follow Python best practices with specific guidelines:

**General Principles**:
- Write clean, readable, self-documenting code
- Follow PEP 8 style guidelines
- Use type hints throughout
- Implement comprehensive error handling
- Write docstrings for all public functions and classes

**Type Hints**:
```python
from typing import List, Dict, Optional, Union, Callable, Awaitable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class AgentConfig:
    """Configuration for agent creation."""
    name: str
    agent_type: str
    capabilities: List[str]
    timeout_seconds: Optional[int] = 60

async def execute_task(
    task: str,
    agent_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> AgentResult:
    """Execute a task using the specified agent."""
    pass
```

**Error Handling**:
```python
class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class AgentNotFoundError(AgentError):
    """Raised when an agent cannot be found."""
    pass

async def get_agent(agent_id: str) -> Agent:
    """Get agent by ID with proper error handling."""
    try:
        agent = AgentRegistry.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent {agent_id} not found")
        return agent
    except Exception as e:
        logger.error(f"Error retrieving agent {agent_id}: {e}")
        raise AgentError(f"Failed to retrieve agent: {e}") from e
```

**Docstring Standards**:
```python
async def orchestrate_task(
    task: str,
    strategy: OrchestrationStrategy,
    requirements: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    priority: TaskPriority = TaskPriority.MEDIUM
) -> OrchestrationResult:
    """
    Orchestrate task execution across multiple agents.
    
    Args:
        task: The task description to execute
        strategy: The orchestration strategy to use
        requirements: Optional list of required agent capabilities
        context: Optional context information for task execution
        priority: Task priority level for scheduling
        
    Returns:
        OrchestrationResult containing task results and metadata
        
    Raises:
        OrchestrationError: If task orchestration fails
        AgentNotFoundError: If no suitable agents are available
        
    Example:
        >>> result = await orchestrate_task(
        ...     task="Research renewable energy trends",
        ...     strategy=OrchestrationStrategy.CONSENSUS,
        ...     requirements=["research", "analysis"]
        ... )
        >>> print(result.primary_result)
    """
```

### Naming Conventions

**Files and Modules**:
- Use lowercase with underscores: `llm_agent.py`, `config_manager.py`
- Module names should be descriptive and clear

**Classes**:
- Use PascalCase: `AgentOrchestrator`, `ConfigManager`
- Abstract classes should end with "Base": `AgentBase`
- Interfaces should end with "Interface": `AgentInterface`

**Functions and Variables**:
- Use snake_case: `execute_task()`, `agent_config`
- Boolean variables should be descriptive: `is_active`, `has_capability`
- Constants should be UPPER_CASE: `MAX_RETRY_ATTEMPTS`

**Configuration Keys**:
- Use lowercase with underscores: `google_api_key`, `max_concurrent_tasks`
- Group related settings: `logging_level`, `logging_format`

## 🧪 Testing Standards

### Testing Philosophy

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Ensure acceptable performance characteristics

### Test Structure

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.agents import AgentFactory, AgentOrchestrator
from src.core.config import Config

class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide mock configuration for tests."""
        return Config(
            environment="test",
            log_level="DEBUG",
            max_concurrent_tasks=5
        )
    
    @pytest.fixture
    async def orchestrator(self, mock_config):
        """Provide orchestrator instance for tests."""
        return AgentOrchestrator(config=mock_config)
    
    @pytest.mark.asyncio
    async def test_single_agent_orchestration(self, orchestrator):
        """Test orchestration with a single agent."""
        # Arrange
        task = "Test task"
        strategy = OrchestrationStrategy.SINGLE_BEST
        
        # Act
        result = await orchestrator.orchestrate_task(task, strategy)
        
        # Assert
        assert result.success
        assert result.primary_result is not None
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_orchestration_with_invalid_strategy(self, orchestrator):
        """Test error handling for invalid strategies."""
        with pytest.raises(ValueError, match="Invalid orchestration strategy"):
            await orchestrator.orchestrate_task("task", "invalid_strategy")
    
    @pytest.mark.parametrize("strategy", [
        OrchestrationStrategy.SINGLE_BEST,
        OrchestrationStrategy.CONSENSUS,
        OrchestrationStrategy.PARALLEL_ALL
    ])
    @pytest.mark.asyncio
    async def test_different_strategies(self, orchestrator, strategy):
        """Test orchestration with different strategies."""
        result = await orchestrator.orchestrate_task("test task", strategy)
        assert result.strategy_used == strategy
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_agents.py

# Run tests with specific markers
python -m pytest -m "unit"
python -m pytest -m "integration"

# Run tests in parallel
python -m pytest -n auto
```

### Test Markers

```python
# Mark test types
@pytest.mark.unit
def test_agent_creation():
    pass

@pytest.mark.integration
async def test_agent_orchestration():
    pass

@pytest.mark.slow
async def test_long_running_task():
    pass

@pytest.mark.requires_api_key
async def test_gemini_integration():
    pass
```

## 🔧 Development Workflow

### Branch Strategy

We follow GitFlow with adaptations for our project:

**Main Branches**:
- `main`: Production-ready code
- `develop`: Integration branch for features

**Feature Branches**:
- `feature/agent-improvements`: New agent functionality
- `feature/ui-enhancements`: User interface changes
- `feature/performance-optimization`: Performance improvements

**Hotfix Branches**:
- `hotfix/critical-bug-fix`: Emergency fixes to production

### Commit Standards

We use Conventional Commits for clear commit messages:

```bash
# Feature commits
feat(agents): add new research agent capabilities
feat(web): implement real-time monitoring dashboard

# Bug fixes
fix(orchestrator): resolve race condition in task allocation
fix(config): handle missing environment variables gracefully

# Documentation
docs(api): update API reference for new endpoints
docs(readme): add installation troubleshooting section

# Refactoring
refactor(agents): simplify agent creation factory pattern
refactor(config): consolidate configuration validation

# Performance
perf(orchestrator): optimize agent selection algorithm
perf(web): implement response caching

# Tests
test(agents): add comprehensive orchestrator tests
test(integration): add end-to-end workflow tests
```

### Pull Request Process

1. **Create Feature Branch**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**:
   ```bash
   # Make your changes
   # Run tests locally
   python -m pytest
   
   # Run linting
   flake8 src/
   black src/
   isort src/
   
   # Run type checking
   mypy src/
   ```

3. **Commit Changes**:
   ```bash
   git add .
   git commit -m "feat(component): description of changes"
   ```

4. **Push and Create PR**:
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

## 🏗️ Architecture Patterns

### Agent Design Patterns

**Abstract Base Pattern**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Agent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.is_active = True
    
    @abstractmethod
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute a task and return the result."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
```

**Factory Pattern**:
```python
class AgentFactory:
    """Factory for creating different types of agents."""
    
    def create_agent(self, agent_type: str, config: Dict[str, Any]) -> Agent:
        """Create agent based on type and configuration."""
        creators = {
            "llm": self._create_llm_agent,
            "workflow": self._create_workflow_agent,
            "custom": self._create_custom_agent
        }
        
        creator = creators.get(agent_type)
        if not creator:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return creator(config)
```

**Observer Pattern for Events**:
```python
from typing import Callable, List

class AgentEventManager:
    """Manage agent events and notifications."""
    
    def __init__(self):
        self._observers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to agent events."""
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(callback)
    
    async def notify(self, event_type: str, data: Dict[str, Any]):
        """Notify all observers of an event."""
        for callback in self._observers.get(event_type, []):
            await callback(data)
```

### Configuration Management

**Environment-Based Configuration**:
```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    """Application configuration with environment overrides."""
    
    # Core settings
    environment: str = "development"
    log_level: str = "INFO"
    
    # Agent settings
    max_concurrent_agents: int = 10
    default_timeout_seconds: int = 60
    
    # API settings
    google_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_concurrent_agents=int(os.getenv("MAX_CONCURRENT_AGENTS", "10")),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY")
        )
```

### Error Handling Patterns

**Hierarchical Exception Structure**:
```python
class PlatformError(Exception):
    """Base exception for all platform errors."""
    pass

class AgentError(PlatformError):
    """Base exception for agent-related errors."""
    pass

class ConfigurationError(PlatformError):
    """Raised for configuration-related issues."""
    pass

class OrchestrationError(AgentError):
    """Raised for orchestration failures."""
    pass
```

**Retry Pattern with Exponential Backoff**:
```python
import asyncio
from typing import Callable, TypeVar, Any

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> T:
    """Retry function with exponential backoff."""
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries:
                raise
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
```

## 📦 Dependency Management

### Adding New Dependencies

1. **Evaluate Necessity**: Ensure the dependency is truly needed
2. **Check Compatibility**: Verify compatibility with existing packages
3. **Add to pyproject.toml**: Use appropriate version constraints

```toml
[project]
dependencies = [
    "fastapi>=0.104.0,<1.0.0",
    "uvicorn[standard]>=0.24.0,<1.0.0",
    "pydantic>=2.5.0,<3.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.7.0"
]
```

### Security Considerations

- Regular dependency updates
- Security vulnerability scanning
- Minimal dependency principle
- Pin critical dependencies

## 🚀 Release Process

### Version Management

We use Semantic Versioning (semver):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Pre-Release**:
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] Performance benchmarks run
   - [ ] Security scan completed

2. **Release**:
   - [ ] Update version number
   - [ ] Create release notes
   - [ ] Tag release in Git
   - [ ] Build and test packages

3. **Post-Release**:
   - [ ] Deploy to staging
   - [ ] Run integration tests
   - [ ] Deploy to production
   - [ ] Monitor for issues

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on technical merit

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Documentation**: Comprehensive guides and references
- **Examples**: Working code samples

### Contributing Types

**Code Contributions**:
- New features
- Bug fixes
- Performance improvements
- Refactoring

**Non-Code Contributions**:
- Documentation improvements
- Issue triaging
- Testing
- User support

### Recognition

Contributors are recognized through:
- GitHub contributor graphs
- Release notes acknowledgments
- Hall of fame documentation
- Maintainer invitations for significant contributors

---

Thank you for contributing to the Multi-Agent Research Platform! Your efforts help make this project better for everyone.